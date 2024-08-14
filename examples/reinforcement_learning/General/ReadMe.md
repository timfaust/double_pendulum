1. start interactive session
``
srun -t 24:00:00 -c 1 --mem 16G -p stud --gres=gpu:1  --pty bash
``
2. install x11
3. adjust x11 permission on local and remote machine. Better ask chat gpt
4. Get a random free port (cluster)

``PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo "Port: $PORT"
echo "Hostname $(hostname)"
``

5. Run an SSH server at that port (cluster)

``/usr/sbin/sshd -D -p $PORT -h ~/.ssh/id_rsa -f ~/.ssh/sshd_config``

6. aktivate tunneling with x11

``
ssh -L 10022:[$(hostname)]:[$PORT] -X [user]@mn.ias.informatik.tu-darmstadt.de
``
7.``ssh -p 10022 -X stud_maraqten@localhost``

#(testen auf local machine)


8. Tensorboard

auf local machine:
''ssh -L 16006:127.0.0.1:6006 stud_maraqten@mn.ias.informatik.tu-darmstadt.de
''

auf cluster: 
zu General folder gehen:

tensorboard --logdir log_data    (es muss mit tab autocpmplete gehem)


dann auf local machine:
''http://127.0.0.1:16006''