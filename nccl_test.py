import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    hostname = os.uname().nodename
    device = torch.cuda.get_device_name(local_rank)
    print(f"[rank {rank}/{world_size}] host={hostname} gpu={device}", flush=True)

    # All-reduce a tensor so we actually move bytes across the LAN
    t = torch.ones(1, device=f"cuda:{local_rank}") * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] all_reduce result = {t.item()} (expected {sum(range(1, world_size + 1))})", flush=True)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
