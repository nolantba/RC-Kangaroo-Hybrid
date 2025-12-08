// ============================================================================
// RCKangaroo Work File Implementation
// Save/Resume/Merge functionality for long-running puzzles
// ============================================================================

#include "WorkFile.h"
#include "XorFilter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <set>

// CRC32 lookup table for checksums
static uint32_t crc32_table[256];
static bool crc32_table_initialized = false;

static void InitCRC32Table() {
    if (crc32_table_initialized) return;

    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = true;
}

// ============================================================================
// RCWorkFile Implementation
// ============================================================================

RCWorkFile::RCWorkFile() : is_loaded(false) {
    InitCRC32Table();
    memset(&header, 0, sizeof(header));
}

RCWorkFile::RCWorkFile(const std::string& fname) : filename(fname), is_loaded(false) {
    InitCRC32Table();
    memset(&header, 0, sizeof(header));
}

RCWorkFile::~RCWorkFile() {
    // Auto-save on destruction if loaded and modified
    if (is_loaded && !filename.empty()) {
        Save();
    }
}

uint32_t RCWorkFile::CalculateChecksum(const void* data, size_t len) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFF;

    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ bytes[i]) & 0xFF];
    }

    return ~crc;
}

bool RCWorkFile::ValidateHeader() {
    // Check magic number
    if (header.magic != WORK_FILE_MAGIC) {
        printf("ERROR: Invalid work file magic (0x%08X, expected 0x%08X)\n",
               header.magic, WORK_FILE_MAGIC);
        return false;
    }

    // Check version
    if (header.version > WORK_FILE_VERSION) {
        printf("ERROR: Work file version %d not supported (max %d)\n",
               header.version, WORK_FILE_VERSION);
        return false;
    }

    // Verify checksum
    uint32_t stored_checksum = header.header_checksum;
    header.header_checksum = 0;  // Zero out for calculation
    uint32_t calculated_checksum = CalculateChecksum(&header, sizeof(header));
    header.header_checksum = stored_checksum;

    if (stored_checksum != calculated_checksum) {
        printf("ERROR: Work file header checksum mismatch (0x%08X != 0x%08X)\n",
               stored_checksum, calculated_checksum);
        return false;
    }

    return true;
}

bool RCWorkFile::Create(uint32_t range_bits, uint32_t dp_bits,
                       const uint8_t* pubkey_x, const uint8_t* pubkey_y,
                       const uint64_t* range_start, const uint64_t* range_stop) {
    // Initialize header
    memset(&header, 0, sizeof(header));

    header.magic = WORK_FILE_MAGIC;
    header.version = WORK_FILE_VERSION;
    header.range_bits = range_bits;
    header.dp_bits = dp_bits;

    // Copy pubkey
    memcpy(header.pubkey_x, pubkey_x, 32);
    memcpy(header.pubkey_y, pubkey_y, 32);

    // Copy range
    if (range_start) {
        memcpy(header.range_start, range_start, 32);
    } else {
        memset(header.range_start, 0, 32);
    }

    if (range_stop) {
        memcpy(header.range_stop, range_stop, 32);
    } else {
        memset(header.range_stop, 0, 32);
    }

    // Initialize progress
    header.total_ops = 0;
    header.dp_count = 0;
    header.dead_kangaroos = 0;
    header.start_time = time(NULL);
    header.last_save_time = header.start_time;
    header.elapsed_seconds = 0;

    // Calculate DP mask
    header.dp_mask_bits = dp_bits;

    // Calculate checksum
    header.header_checksum = 0;
    header.header_checksum = CalculateChecksum(&header, sizeof(header));

    is_loaded = true;
    dp_records.clear();

    return true;
}

bool RCWorkFile::Save() {
    if (filename.empty()) {
        printf("ERROR: No filename specified for save\n");
        return false;
    }

    return SaveAs(filename);
}

bool RCWorkFile::SaveAs(const std::string& new_filename) {
    // Atomic write: write to temp file, then rename
    std::string temp_filename = new_filename + ".tmp";

    FILE* f = fopen(temp_filename.c_str(), "wb");
    if (!f) {
        printf("ERROR: Cannot create work file: %s\n", temp_filename.c_str());
        return false;
    }

    // Update timestamps
    header.last_save_time = time(NULL);
    header.dp_count = dp_records.size();

    // Recalculate checksum
    header.header_checksum = 0;
    header.header_checksum = CalculateChecksum(&header, sizeof(header));

    // Write header
    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        printf("ERROR: Failed to write header\n");
        fclose(f);
        remove(temp_filename.c_str());
        return false;
    }

    // Write DP records
    if (!dp_records.empty()) {
        size_t written = fwrite(dp_records.data(), sizeof(DPRecord),
                               dp_records.size(), f);
        if (written != dp_records.size()) {
            printf("ERROR: Failed to write DP records (%zu/%zu)\n",
                   written, dp_records.size());
            fclose(f);
            remove(temp_filename.c_str());
            return false;
        }
    }

    fclose(f);

    // Atomic rename
    remove(new_filename.c_str());  // Remove old file
    if (rename(temp_filename.c_str(), new_filename.c_str()) != 0) {
        printf("ERROR: Failed to rename temp file\n");
        remove(temp_filename.c_str());
        return false;
    }

    filename = new_filename;

    printf("Work file saved: %s (%llu ops, %llu DPs)\n",
           filename.c_str(), header.total_ops, header.dp_count);

    return true;
}

bool RCWorkFile::Load() {
    if (filename.empty()) {
        printf("ERROR: No filename specified for load\n");
        return false;
    }

    return Load(filename);
}

bool RCWorkFile::Load(const std::string& fname) {
    FILE* f = fopen(fname.c_str(), "rb");
    if (!f) {
        printf("ERROR: Cannot open work file: %s\n", fname.c_str());
        return false;
    }

    // Read header
    if (fread(&header, sizeof(header), 1, f) != 1) {
        printf("ERROR: Failed to read header\n");
        fclose(f);
        return false;
    }

    // Validate header
    if (!ValidateHeader()) {
        fclose(f);
        return false;
    }

    // Read DP records
    dp_records.clear();
    if (header.dp_count > 0) {
        dp_records.resize(header.dp_count);

        size_t read_count = fread(dp_records.data(), sizeof(DPRecord),
                                 header.dp_count, f);
        if (read_count != header.dp_count) {
            printf("ERROR: Failed to read DP records (%zu/%llu)\n",
                   read_count, header.dp_count);
            fclose(f);
            return false;
        }
    }

    fclose(f);

    filename = fname;
    is_loaded = true;

    printf("Work file loaded: %s\n", filename.c_str());
    printf("  Range: %u bits, DP: %u bits\n", header.range_bits, header.dp_bits);
    printf("  Progress: %llu ops, %llu DPs\n", header.total_ops, header.dp_count);
    printf("  Elapsed: %llu seconds (%.1f hours)\n",
           header.elapsed_seconds, header.elapsed_seconds / 3600.0);

    return true;
}

bool RCWorkFile::AddDP(const uint8_t* dp_x, const uint8_t* distance, uint8_t type) {
    if (!is_loaded) {
        printf("ERROR: Work file not initialized\n");
        return false;
    }

    // No duplicate check needed here - caller already verified DP is unique
    // via database FindOrAddDataBlock check before calling AddDP

    // Add new DP
    DPRecord record;
    memcpy(record.dp_x, dp_x, 12);
    memcpy(record.distance, distance, 22);
    record.type = type;
    record.reserved = 0;

    dp_records.push_back(record);
    header.dp_count = dp_records.size();

    return true;
}

void RCWorkFile::UpdateProgress(uint64_t ops, uint64_t dps,
                                uint64_t dead, uint64_t elapsed) {
    header.total_ops = ops;
    header.dp_count = dps;
    header.dead_kangaroos = dead;
    header.elapsed_seconds = elapsed;
    header.last_save_time = time(NULL);
}

bool RCWorkFile::HasDP(const uint8_t* dp_x) const {
    for (const auto& dp : dp_records) {
        if (memcmp(dp.dp_x, dp_x, 12) == 0) {
            return true;
        }
    }
    return false;
}

bool RCWorkFile::Merge(const std::vector<std::string>& input_files,
                      const std::string& output_file) {
    if (input_files.empty()) {
        printf("ERROR: No input files specified for merge\n");
        return false;
    }

    printf("Merging %zu work files...\n", input_files.size());

    // Load first file as base
    RCWorkFile merged;
    if (!merged.Load(input_files[0])) {
        printf("ERROR: Failed to load base file: %s\n", input_files[0].c_str());
        return false;
    }

    printf("Base file: %s (%llu ops, %llu DPs)\n",
           input_files[0].c_str(), merged.header.total_ops, merged.header.dp_count);

    // Build XOR filter from base DPs for fast duplicate detection
    printf("Building XOR filter for duplicate detection...\n");
    DPXorFilter dp_filter;
    std::vector<uint8_t> dp_data;
    for (const auto& dp : merged.dp_records) {
        dp_data.insert(dp_data.end(), dp.dp_x, dp.dp_x + 12);
    }
    if (!dp_filter.BuildFromDPs(dp_data)) {
        printf("WARNING: Failed to build XOR filter, using slow duplicate detection\n");
    } else {
        printf("XOR filter built: %.1f MB for %llu DPs\n",
               dp_filter.GetSizeBytes() / 1024.0 / 1024.0, merged.header.dp_count);
    }

    // Merge additional files
    uint64_t total_ops = merged.header.total_ops;
    uint64_t total_elapsed = merged.header.elapsed_seconds;
    uint64_t total_dead = merged.header.dead_kangaroos;
    uint64_t duplicates_found = 0;

    for (size_t i = 1; i < input_files.size(); i++) {
        printf("\nMerging file %zu/%zu: %s\n", i+1, input_files.size(),
               input_files[i].c_str());

        RCWorkFile work;
        if (!work.Load(input_files[i])) {
            printf("WARNING: Failed to load file, skipping: %s\n",
                   input_files[i].c_str());
            continue;
        }

        // Verify compatibility
        if (!merged.IsCompatible(work.header.range_bits, work.header.dp_bits,
                                work.header.pubkey_x, work.header.pubkey_y)) {
            printf("ERROR: Incompatible work file: %s\n", input_files[i].c_str());
            return false;
        }

        // Verify RNG seed compatibility for distance integrity
        // If either file has rng_seed=0 (mixed/unknown), allow merge but warn
        // If both have non-zero seeds but they differ, that's an error
        if (merged.header.rng_seed != 0 && work.header.rng_seed != 0 &&
            merged.header.rng_seed != work.header.rng_seed) {
            printf("ERROR: RNG seed mismatch - cannot merge files with different kangaroo seeds\n");
            printf("  Base file seed: %llu\n", (unsigned long long)merged.header.rng_seed);
            printf("  This file seed: %llu\n", (unsigned long long)work.header.rng_seed);
            printf("  (Merging files with different seeds causes distance incompatibility)\n");
            return false;
        }
        if (work.header.rng_seed == 0 && merged.header.rng_seed != 0) {
            printf("WARNING: Merging file with unknown RNG seed (rng_seed=0)\n");
        }

        // Add ops and elapsed time
        total_ops += work.header.total_ops;
        total_elapsed += work.header.elapsed_seconds;
        total_dead += work.header.dead_kangaroos;

        // Merge DPs (skip duplicates)
        size_t added = 0;
        for (const auto& dp : work.dp_records) {
            // Fast duplicate check with XOR filter
            bool is_duplicate = false;
            if (dp_filter.IsBuilt()) {
                is_duplicate = dp_filter.ContainsDP(dp.dp_x);
            } else {
                // Fallback to linear search
                is_duplicate = merged.HasDP(dp.dp_x);
            }

            if (!is_duplicate) {
                merged.dp_records.push_back(dp);
                added++;
            } else {
                duplicates_found++;
            }
        }

        printf("  Added %zu DPs, skipped %llu duplicates\n", added, duplicates_found);
        printf("  Total: %llu ops, %zu DPs\n", total_ops, merged.dp_records.size());
    }

    // Update merged header
    merged.header.total_ops = total_ops;
    merged.header.dp_count = merged.dp_records.size();
    merged.header.elapsed_seconds = total_elapsed;
    merged.header.dead_kangaroos = total_dead;
    // Note: rng_seed is preserved from base file (or remains 0 if base had 0)

    // Save merged file
    printf("\nSaving merged file: %s\n", output_file.c_str());
    if (!merged.SaveAs(output_file)) {
        printf("ERROR: Failed to save merged file\n");
        return false;
    }

    printf("\nMerge complete!\n");
    printf("  Input files: %zu\n", input_files.size());
    printf("  Total operations: %llu\n", total_ops);
    printf("  Total DPs: %llu\n", merged.header.dp_count);
    printf("  Duplicates removed: %llu\n", duplicates_found);
    printf("  Total elapsed: %.1f hours\n", total_elapsed / 3600.0);
    if (merged.header.rng_seed != 0) {
        printf("  RNG seed: %llu (resumable with same seed)\n",
               (unsigned long long)merged.header.rng_seed);
    } else {
        printf("  RNG seed: 0 (mixed sources, resume not recommended)\n");
    }

    return true;
}

bool RCWorkFile::VerifyIntegrity() {
    if (!is_loaded) {
        printf("ERROR: Work file not loaded\n");
        return false;
    }

    // Verify header
    if (!ValidateHeader()) {
        return false;
    }

    // Verify DP count matches
    if (header.dp_count != dp_records.size()) {
        printf("ERROR: DP count mismatch (%llu in header, %zu in records)\n",
               header.dp_count, dp_records.size());
        return false;
    }

    printf("Work file integrity verified: %s\n", filename.c_str());
    return true;
}

bool RCWorkFile::IsCompatible(uint32_t range_bits, uint32_t dp_bits,
                             const uint8_t* pubkey_x, const uint8_t* pubkey_y) {
    if (header.range_bits != range_bits) {
        printf("ERROR: Range mismatch (%u != %u)\n", header.range_bits, range_bits);
        return false;
    }

    if (header.dp_bits != dp_bits) {
        printf("ERROR: DP bits mismatch (%u != %u)\n", header.dp_bits, dp_bits);
        return false;
    }

    if (memcmp(header.pubkey_x, pubkey_x, 32) != 0) {
        printf("ERROR: Public key X mismatch\n");
        return false;
    }

    if (memcmp(header.pubkey_y, pubkey_y, 32) != 0) {
        printf("ERROR: Public key Y mismatch\n");
        return false;
    }

    return true;
}

void RCWorkFile::PrintInfo() const {
    printf("=== Work File Info ===\n");
    printf("File: %s\n", filename.c_str());
    printf("Range: %u bits\n", header.range_bits);
    printf("DP bits: %u\n", header.dp_bits);
    printf("Operations: %llu (2^%.2f)\n", header.total_ops,
           log2((double)header.total_ops));
    printf("DPs found: %llu\n", header.dp_count);
    printf("Dead kangaroos: %llu\n", header.dead_kangaroos);

    uint64_t elapsed = header.elapsed_seconds;
    uint64_t hours = elapsed / 3600;
    uint64_t minutes = (elapsed % 3600) / 60;
    uint64_t seconds = elapsed % 60;
    printf("Elapsed: %02llu:%02llu:%02llu\n", hours, minutes, seconds);

    time_t start_time = header.start_time;
    printf("Started: %s", ctime(&start_time));

    time_t last_save = header.last_save_time;
    printf("Last saved: %s", ctime(&last_save));
}

std::string RCWorkFile::GetInfoString() const {
    char buf[1024];
    snprintf(buf, sizeof(buf),
             "Range:%u DP:%u Ops:%llu DPs:%llu Elapsed:%llus",
             header.range_bits, header.dp_bits, header.total_ops,
             header.dp_count, header.elapsed_seconds);
    return std::string(buf);
}

// ============================================================================
// AutoSaveManager Implementation
// ============================================================================

AutoSaveManager::AutoSaveManager(RCWorkFile* wf, uint64_t interval_sec)
    : work_file(wf), save_interval_seconds(interval_sec),
      last_save_time(0), enabled(true) {
}

bool AutoSaveManager::CheckAndSave(uint64_t current_ops, uint64_t current_dps,
                                  uint64_t dead_kangaroos, uint64_t elapsed_sec) {
    if (!enabled || !work_file) {
        return false;
    }

    // Check if it's time to save
    uint64_t current_time = time(NULL);
    if (current_time - last_save_time >= save_interval_seconds) {
        return ForceSave(current_ops, current_dps, dead_kangaroos, elapsed_sec);
    }

    return false;
}

bool AutoSaveManager::ForceSave(uint64_t current_ops, uint64_t current_dps,
                               uint64_t dead_kangaroos, uint64_t elapsed_sec) {
    if (!work_file) {
        return false;
    }

    // Update progress
    work_file->UpdateProgress(current_ops, current_dps, dead_kangaroos, elapsed_sec);

    // Save
    bool result = work_file->Save();

    if (result) {
        last_save_time = time(NULL);
    }

    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string GenerateWorkFilename(uint32_t range_bits, const uint8_t* pubkey_x) {
    char filename[256];
    snprintf(filename, sizeof(filename), "puzzle%u_%02x%02x%02x%02x.work",
             range_bits, pubkey_x[0], pubkey_x[1], pubkey_x[2], pubkey_x[3]);
    return std::string(filename);
}

bool WorkFileExists(const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

bool GetWorkFileInfo(const std::string& filename, WorkFileHeader* header_out) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) return false;

    bool result = (fread(header_out, sizeof(WorkFileHeader), 1, f) == 1);
    fclose(f);

    return result;
}
