# Source this: gives pod_ssh for the Avaz training pod, with the test-only key and its own known_hosts file.
SECRETS=<folder with the pod ssh key>
POD_ID=<pod id>
POD_HOST=root@<pod ip>
POD_PORT=<ssh port>
SSH_OPTS=(-i "$SECRETS/pod_key" -p "$POD_PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=<known_hosts file> -o ConnectTimeout=15 -o ServerAliveInterval=30 -o BatchMode=yes)
pod_ssh() { ssh "${SSH_OPTS[@]}" "$POD_HOST" "$@"; }
