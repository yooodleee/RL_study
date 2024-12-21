import argparse
import os
import sys
import six.moves import shlex_quote


parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=20, type=int, help="Number of workers")
parser.add_argument(
    '-r', '--romotes', default=None, help='The address of pre-existing VNC servers and rewards to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).'
)
parser.add_argument('-e', '--env-id', type=str, default="doom", help="Environment id")
parser.add_argument('-l', '--log-dir', type=str, default="tmp/doom", help="Log directory path")
parser.add_argument('-n', '--dry-run', action='store_true', help="Print out commands rather than executing them")
parser.add_argument('-m', '--mode', type=str, default='tmux', help="tmux: run workers in a tmux session. hohup: run worksers with nohup. child: run workers as child processes")
parser.add_argument('--visualise', action='store_true', help="Visualise the gym environment by running env.render() between each timestep")
parser.add_argument('--envWrap', action='store_true', help="Preprocess input in env_wrapper (no change in input size or network)")
parser.add_argument('--designHead', type=str, default='universe', help="Network design head: nips or nature or doom or universe(default)")
parser.add_argument('--unsup', type=str, default=None, help="Unsup. exploration mode: action or state or stateAenc or None")
parser.add_argument('--noReward', action='store_true', help="Remove all extrinsic reward")
parser.add_argument('--noLifeReward', action='store_true', help="Remove all negative reward (in doom: it is living reward)")
parser.add_argument('--expName', type=str, default='a3c', help="Experiment tmux session-name. Default a3c.")
parser.add_argument('--expId', type=int, default=0, help="Experiment Id >= 0. Needed while running more than one run per machine.")
parser.add_argument('--savio', action='store_true', help="Savio or KNL cpu cluster hacks")
parser.add_argument('--default', action='store_true', help="run with default params")
parser.add_argument('--pretrain', type=str, default=None, help="Checkpoint dir (generally .../train/) to load from.")

def new_cmd(session, name, cmd, mode, logdir, shell):
    if isinstance(cmd, (list, tuple)):
        cmd == " ".join(shlex_quote(str(v)) for v in cmd)
    if mode == 'tmux':
        return name, "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd))
    elif mode == 'child':
        return name, "{} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(cmd, logdir, session, name, logdir)
    elif mode == 'nohup':
        return name, "nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(shell, shlex_quote(cmd), logdir, session, name, logdir)


def create_commands(
    session,
    num_workers,
    remotes,
    env_id,
    logdir,
    shell='bash',
    mode='tmux',
    visualise=False,
    envWrap=False,
    designHead=None,
    unsup=None,
    noReward=False,
    noLifeReward=False,
    psPort=12222,
    delay=0,
    savio=False,
    pretrain=None
):
    # for launching the TF workers and for launching tensorboard
    py_cmd = 'python' if savio else sys.executable
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=',
        py_cmd, 'worker.py',
        '--log-dir', logdir,
        '--env-id', env_id,
        '--num-workers', str(num_workers),
        '--psPort', psPort]
    
    if delay > 0:
        base_cmd += ['--delay', delay]
    if visualise:
        base_cmd += ['--visualise']
    if envWrap:
        base_cmd += ['--envWrap']
    if designHead is not None:
        base_cmd += ['--designHead', designHead]
    if unsup is not None:
        base_cmd += ['--unsup', unsup]
    if noReward:
        base_cmd += ['--noReward']
    if noLifeReward:
        base_cmd += ['--noLifeReward']
    if pretrain is not None:
        base_cmd += ['--pretrain', pretrain]
    
    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers
    
    cmds_map = [new_cmd(session, "ps", base_cmd + ["--job-name", "ps"], mode, logdir, shell)]
    for i in range(num_workers):
        cmds_map += [new_cmd(session, "w-%d" % i, base_cmd + ["--job-name", "worker", "--task", str(i), "--remotes", remotes[i]], mode, logdir, shell)]
    
    # No tensorboard or htop window if running multiple experiments per machine
    if session == 'a3c':
        cmds_map += [new_cmd(session, "tb", ["tensorboard", "--logdir", logdir, "--port", "12345", remotes[i]],  mode, logdir, shell)]
    if session == 'a3c' and mode == 'tmux':
        cmds_map += [new_cmd(session, "htop", ["htop"], mode, logdir, shell)]
    
    windows = [v[0] for v in cmds_map]

    notes = []
    cmds = [
        "mkdir -p {}".format(logdir),
        "echo {} {} > {}/cmd.sh".format(sys.executable, ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']), logdir),
    ]
    