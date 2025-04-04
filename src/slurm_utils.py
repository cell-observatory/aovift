
import logging
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger('')

hostname = "login.abc.berkeley.edu"
username = "thayeralshaabi"


def get_number_of_idle_nodes(partition: str = "abc_a100"):
    retry = True
    while retry:
        table = subprocess.run(
            f"ssh {username}@{hostname} \"sinfo -p {partition} --states idle -O NODES\"",
            capture_output=True,
            shell=True
        )

        response = str(table.stdout)
        error_str = str(table.stderr)
        if 'unbound variable' not in error_str:
            retry = True
            logger.error(f'Retrying because of : {error_str}')
        else:
            retry = False

    try:
        response = response.split("NODES")[1]
        print(f'NODES {response=}')
        number_of_idle_nodes = int(response.split(r"\n")[1])
    except:
        print(f'{response=}')
        number_of_idle_nodes = 0

    print(f'Number of idle nodes is {number_of_idle_nodes} on {partition}.')
    return number_of_idle_nodes


def get_available_resources(requested_partition='abc_a100'):
    resources = {}

    nodes = str(subprocess.run(
        f'ssh {username}@{hostname} "scontrol show nodes"',
        capture_output=True,
        shell=True
    ).stdout)

    node_names = list(map(lambda x: f"{x.replace('NodeName=', '')}.abc0", re.findall(r"NodeName=\w+", str(nodes))))
    node_partitions = list(map(lambda x: x.replace('Partitions=', ''), re.findall(r"Partitions=\w+", str(nodes))))
    configured = list(map(lambda x: int(x.replace('CPUTot=', '')), re.findall(r"CPUTot=\d+", str(nodes))))
    allocated = list(map(lambda x: int(x.replace('CPUAlloc=', '')), re.findall(r"CPUAlloc=\d+", str(nodes))))

    for name, partition, total_cpus, allocated_cpus in zip(node_names, node_partitions, configured, allocated):

        if partition == requested_partition:
            available_cpus = total_cpus - allocated_cpus
            available_gpus = available_cpus // 4 if available_cpus != 0 else 0

            resources[name] = {
                "total_cpus": total_cpus,
                "available_cpus": available_cpus,
                "total_mem": f"{30 * total_cpus}GB",  # 30GB per core
                "available_mem": f"{30 * available_cpus}GB" if available_cpus != 0 else 0,  # 30GB per core
                "total_gpus": total_cpus // 4,  # 4 cores per gpu
                "available_gpus": available_gpus,  # 4 cores per gpu
            }

    return pd.DataFrame.from_dict(resources, orient='index')


def get_active_branch_name(head_dir):
    head_dir = Path(head_dir) / ".git" / "HEAD"
    with head_dir.open("r") as f:
        content = f.read().splitlines()
    
    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


def paths_to_clusterfs(flags:str, local_repo):
    if isinstance(flags, Path):
        flags = str(flags)
        flags_is_path = True
    else:
        flags_is_path = False

    flags = re.sub(pattern="\\\\", repl='/', string=flags)  # regex needs four backslashes to indicate one
    
    if local_repo is not None:
        flags = flags.replace("..", local_repo)  # regex stinks at replacing ".."
    
    flags = re.sub(pattern='/home/supernova/nvme2/', repl='/clusterfs/nvme2/', string=flags)
    flags = re.sub(pattern='~/nvme2', repl='/clusterfs/nvme2/', string=flags)
    flags = re.sub(pattern='U:\\\\', repl='/clusterfs/nvme2/', string=flags)
    flags = re.sub(pattern='U:/', repl='/clusterfs/nvme2/', string=flags)
    flags = re.sub(pattern='V:\\\\', repl='/clusterfs/nvme/', string=flags)
    flags = re.sub(pattern='V:/', repl='/clusterfs/nvme/', string=flags)
    flags = re.sub(pattern='D:/', repl='/d_drive/', string=flags)
    flags = re.sub(pattern='C:/', repl='/c_drive/', string=flags)

    if flags_is_path:
        flags = Path(flags)
    return flags


def submit_slurm_job(args, command_flags, partition: str = "abc_a100"):
    framework = "main_TF"
    CUDA_version = "CUDA_12_3"
    cluster_repo = f"/clusterfs/nvme/thayer/aovift"
    cluster_env = f"apptainer exec --bind /clusterfs --nv {cluster_repo}/{framework}_{CUDA_version}.sif python "
    script = f"{cluster_repo}/src/ao.py"
    
    flags = ' '.join(command_flags)
    flags = re.sub(pattern='--cluster', repl='', string=flags)
    flags = re.sub(pattern='--docker', repl='', string=flags)
    flags = paths_to_clusterfs(flags, cluster_repo)
    
    # available_nodes = slurm_utils.get_available_resources(
    #     username=username,
    #     hostname=hostname,
    #     requested_partition='abc_a100'
    # ).sort_values('available_gpus', ascending=False)
    #
    # print(available_nodes)
    # desired_node = available_nodes.iloc[0].to_dict()
    
    # flags = re.sub(
    #     pattern='--batch_size \d+',  # replace w/ 896; max number of samples we can fit on A100 w/ 80G of vram
    #     repl=f'--batch_size {896*desired_node["available_gpus"]}',
    #     string=flags
    # )
    
    sjob = f"srun "
    sjob += f"-p {partition} "
    sjob += f" --nodes=1 "
    # sjob += f' --gres=gpu:{desired_node["available_gpus"]} '
    # sjob += f' --cpus-per-task={desired_node["available_cpus"]} '
    # sjob += f" --mem='{desired_node['available_mem']}' "
    # sjob += f" --nodelist='{available_nodes.index[0]}' "
    sjob += f"--exclusive "
    sjob += f"--job-name={args.func}_{args.input.stem} "
    sjob += f"{cluster_env} {script} {flags}"
    # sjob += f" &> {args.input.parent}/{args.func}_{args.input.stem}.log"
    logger.info(sjob)
    subprocess.run(f"ssh {username}@{hostname} \"{sjob}\"", shell=True)


def submit_docker_job(args=None, command_flags=None) -> int:
    '''

    Args:
        args:
        command_flags:

    Returns:
    return code of the command, unless Docker Run didn't execute and then that will return error code
    '''

    container_repo = "/app/aovift"  # location of repo in the container
    local_repo = Path(__file__).parent.parent  # location of repo in host
    branch_name = get_active_branch_name(local_repo)
    CUDA_version = "tf_cuda_12_3"

    if command_flags is None:
        flags = ' '.join(args)
    else:
        flags = ' '.join(command_flags)

    flags = re.sub(pattern=' --docker', repl='', string=flags)  # remove flag
    flags = paths_to_clusterfs(flags, container_repo)
    flags = re.sub(pattern=local_repo.as_posix(), repl=container_repo, string=flags)
    docker_container_name = 'opt_net'

    docker_run = ("docker run --rm "
                  "--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"   # GPU stuff
                  f" --name {docker_container_name} --privileged=true -u 1000")  # privileged means sudo is available to user
    docker_mount = (f'-v "{local_repo}":{container_repo}  '
                    r'-v D:\:/d_drive  '
                    r'-v C:\:/c_drive  '
                    )
    if os.name == 'nt':
        docker_mount = docker_mount + r'-v %userprofile%/.ssh:/sshkey '
    else:
        docker_mount = docker_mount + r'-v ~/.ssh:/sshkey '

    docker_vars = (r' -e RUNNING_IN_DOCKER=TRUE'
                   r' -e USER=vscode ')
    docker_image = f"ghcr.io/cell-observatory/aovift:{branch_name}_{CUDA_version}"
    docker_job = f'{docker_run} {docker_vars} --workdir {container_repo}/src {docker_mount} {docker_image}    "python ao.py {flags}"'
    docker_remove_old = f'docker rm  --force {docker_container_name} || True'    # kill container if it was orphaned.
    print(f"Docker job: \n{docker_job}\n")
    subprocess.run(docker_remove_old, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)  # supress output
    completedprocess = subprocess.run(docker_job, shell=True)  # docker will return command's error code unless docker fails.
    return completedprocess.returncode
