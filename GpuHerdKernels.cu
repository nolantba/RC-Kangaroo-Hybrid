#include "GpuHerdManager.h"
#include "RCGpuUtils.h"
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================================
// Herd-Aware Kangaroo Kernel
// ============================================================================
// Each thread represents one kangaroo
// Kangaroos are grouped into herds with different jump tables
// ============================================================================

__global__ void kangarooHerdKernel(
    const uint64_t* __restrict__ jump_tables,  // [herds][jump_table_size]
    GpuHerdState* __restrict__ herd_states,
    HerdDPBuffer* __restrict__ herd_buffers,
    DP* __restrict__ gpu_dp_buffer,
    int* __restrict__ gpu_dp_count,
    const HerdConfig config,
    u64* kangaroo_x,     // Kangaroo X coordinates [threads][4]
    u64* kangaroo_y,     // Kangaroo Y coordinates [threads][4]
    u64* kangaroo_dist,  // Kangaroo distances [threads][4]
    int iterations
)
{
    // Calculate herd ID and position within herd
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_kangaroos = config.herds_per_gpu * config.kangaroos_per_herd;
    
    if (global_tid >= total_kangaroos) return;
    
    int herd_id = global_tid / config.kangaroos_per_herd;
    int local_tid = global_tid % config.kangaroos_per_herd;
    
    // Get this herd's jump table
    const uint64_t* my_jumps = jump_tables + (herd_id * config.jump_table_size);
    
    // Load kangaroo state (256-bit values)
    u64 kx[4], ky[4], dist[4];
    LoadU256(kx, kangaroo_x + global_tid * 4);
    LoadU256(ky, kangaroo_y + global_tid * 4);
    LoadU256(dist, kangaroo_dist + global_tid * 4);
    
    // Perform iterations
    for (int iter = 0; iter < iterations; iter++) {
        // Select jump distance using herd-specific table
        // Use bottom bits of X coordinate as index
        int jump_idx = (int)(kx[0] % config.jump_table_size);
        uint64_t jump_dist = my_jumps[jump_idx];
        
        // TODO: Perform elliptic curve point addition here
        // This is where you'd integrate with your existing EC operations
        // Example pseudocode:
        // Point jump_point = precomputed_jumps[jump_idx];
        // AddPointsJacobian(kx, ky, jump_point.x, jump_point.y);
        
        // Update distance
        // Add256(dist, dist, jump_dist);
        
        // Check if distinguished point (using your fixed clz256)
        int lz = clz256(kx);  // Your fixed 256-bit leading zero count
        
        if (lz >= config.dp_bits) {
            // This is a DP!
            
            // Try to add to herd-local buffer first (fast path)
            int local_idx = atomicAdd(&herd_buffers[herd_id].count, 1);
            
            if (local_idx < config.herd_dp_buffer_size) {
                // Space available in local buffer
                int write_pos = (herd_buffers[herd_id].write_idx + local_idx) 
                                % config.herd_dp_buffer_size;
                
                // Create DP
                DP new_dp;
                // Copy X coordinate (tail 96 bits for compact storage)
                for (int i = 0; i < 3; i++) {
                    ((uint32_t*)new_dp.x)[i] = (uint32_t)kx[i];
                }
                
                // Copy distance
                for (int i = 0; i < 4; i++) {
                    ((uint64_t*)new_dp.d)[i] = dist[i];
                }
                
                // Set type (TAME or WILD based on herd_id)
                new_dp.type = (herd_id % 2 == 0) ? TAME : WILD1;
                
                // Store in herd buffer
                herd_buffers[herd_id].dps[write_pos] = new_dp;
                
                // Update herd stats
                atomicAdd((unsigned long long*)&herd_states[herd_id].dps_found, 1ULL);
            } else {
                // Local buffer full, promote to GPU shared buffer (slow path)
                int gpu_idx = atomicAdd(gpu_dp_count, 1);
                if (gpu_idx < config.gpu_dp_buffer_size) {
                    DP new_dp;
                    // ... fill DP structure same as above ...
                    gpu_dp_buffer[gpu_idx] = new_dp;
                }
            }
        }
        
        // Update herd operation counter (one thread per herd)
        if (local_tid == 0 && iter % 100 == 0) {
            atomicAdd((unsigned long long*)&herd_states[herd_id].operations, 
                     (unsigned long long)(config.kangaroos_per_herd * 100));
        }
    }
    
    // Store kangaroo state back to global memory
    StoreU256(kangaroo_x + global_tid * 4, kx);
    StoreU256(kangaroo_y + global_tid * 4, ky);
    StoreU256(kangaroo_dist + global_tid * 4, dist);
}

// ============================================================================
// Host Function: Launch Herd Kernels
// ============================================================================

extern "C" void launchHerdKernels(
    GpuHerdMemory* mem,
    u64* d_kangaroo_x,
    u64* d_kangaroo_y,
    u64* d_kangaroo_dist,
    int iterations
)
{
    int num_blocks = mem->config.getNumBlocks();
    int threads_per_block = mem->config.threads_per_block;
    
    // Launch kernel
    kangarooHerdKernel<<<num_blocks, threads_per_block>>>(
        mem->d_jump_tables,
        mem->d_herd_states,
        mem->d_herd_buffers,
        mem->d_gpu_dp_buffer,
        mem->d_gpu_dp_count,
        mem->config,
        d_kangaroo_x,
        d_kangaroo_y,
        d_kangaroo_dist,
        iterations
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// Host Function: Check for Collisions
// ============================================================================

extern "C" int checkHerdCollisions(
    GpuHerdMemory* mem,
    DP* host_dp_buffer,
    int max_dps
)
{
    // Copy GPU DP buffer to host
    int dp_count;
    cudaMemcpy(&dp_count, mem->d_gpu_dp_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (dp_count > max_dps) dp_count = max_dps;
    
    if (dp_count > 0) {
        cudaMemcpy(host_dp_buffer, mem->d_gpu_dp_buffer, 
                   dp_count * sizeof(DP), cudaMemcpyDeviceToHost);
        
        // Reset GPU buffer
        cudaMemset(mem->d_gpu_dp_count, 0, sizeof(int));
    }
    
    return dp_count;
}
