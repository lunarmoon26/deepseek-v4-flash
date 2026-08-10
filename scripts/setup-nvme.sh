#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env"
fi

if (( EUID != 0 )); then
  printf '%s\n' "Run with: sudo env ERASE_DEEPSEEK_NVME=YES ./scripts/setup-nvme.sh"
  exit 1
fi
if [[ "${ERASE_DEEPSEEK_NVME:-}" != "YES" ]]; then
  printf '%s\n' "Refusing to erase disks without ERASE_DEEPSEEK_NVME=YES."
  exit 1
fi

# Logical drive order is tied to the physical serials below. Kernel NVMe
# device numbers can change after a reboot, so resolve /dev paths at runtime.
for variable in DEEPSEEK_NVME1_SERIAL DEEPSEEK_NVME2_SERIAL; do
  [[ -n "${!variable:-}" ]] || {
    printf 'Set %s in the ignored .env file before provisioning disks.\n' \
      "$variable"
    exit 1
  }
done
serials=("$DEEPSEEK_NVME1_SERIAL" "$DEEPSEEK_NVME2_SERIAL")
mounts=(
  "${DEEPSEEK_NVME1_MOUNT:-/mnt/deepseek-nvme1}"
  "${DEEPSEEK_NVME2_MOUNT:-/mnt/deepseek-nvme2}"
)
data_labels=(dsv4-nvme1 dsv4-nvme2)
swap_labels=(dsv4-swap1 dsv4-swap2)

for command in lsblk parted partprobe wipefs mkfs.ext4 mkswap swapon blkid udevadm; do
  command -v "$command" >/dev/null || {
    printf 'Missing required command: %s\n' "$command"
    exit 1
  }
done

devices=()
for expected_serial in "${serials[@]}"; do
  resolved_device=""
  while read -r name actual_serial type; do
    if [[ "$type" == "disk" && "$actual_serial" == "$expected_serial" ]]; then
      [[ -z "$resolved_device" ]] || {
        printf 'Serial %s matched more than one disk.\n' "$expected_serial"
        exit 1
      }
      resolved_device="$name"
    fi
  done < <(lsblk -dnpo NAME,SERIAL,TYPE)
  [[ -n "$resolved_device" ]] || {
    printf 'Could not find the disk with serial %s.\n' "$expected_serial"
    exit 1
  }
  devices+=("$resolved_device")
done

for index in 0 1; do
  device="${devices[$index]}"
  [[ -b "$device" ]] || {
    printf 'Missing block device: %s\n' "$device"
    exit 1
  }
  actual_serial="$(lsblk -dn -o SERIAL "$device" | tr -d '[:space:]')"
  [[ "$actual_serial" == "${serials[$index]}" ]] || {
    printf 'Serial mismatch for %s: expected %s, found %s\n' \
      "$device" "${serials[$index]}" "$actual_serial"
    exit 1
  }
  while read -r mountpoint; do
    [[ -z "$mountpoint" ]] || {
      printf 'Refusing to erase mounted device %s (%s).\n' "$device" "$mountpoint"
      exit 1
    }
  done < <(lsblk -nr -o MOUNTPOINTS "$device")
done

printf '%s\n' "Erasing both confirmed SN8100 SSDs and creating swap + ext4 partitions..."
for index in 0 1; do
  device="${devices[$index]}"
  for partition in "${device}"p1 "${device}"p2; do
    [[ -b "$partition" ]] && wipefs --all --force "$partition"
  done
  wipefs --all --force "$device"
  parted --script "$device" mklabel gpt
  parted --script "$device" mkpart primary linux-swap 1MiB 129GiB
  parted --script "$device" mkpart primary ext4 129GiB 100%
done

partprobe "${devices[@]}"
udevadm settle

for index in 0 1; do
  device="${devices[$index]}"
  mkswap --force --label "${swap_labels[$index]}" "${device}p1"
  mkfs.ext4 -F -m 0 -L "${data_labels[$index]}" "${device}p2"
  mkdir -p "${mounts[$index]}"
done

line_exists() {
  local expected="$1"
  local line
  while IFS= read -r line; do
    [[ "$line" == "$expected" ]] && return 0
  done < /etc/fstab
  return 1
}

for index in 0 1; do
  swap_uuid="$(blkid -s UUID -o value "${devices[$index]}p1")"
  data_uuid="$(blkid -s UUID -o value "${devices[$index]}p2")"
  swap_line="UUID=$swap_uuid none swap defaults,pri=100,nofail 0 0"
  data_line="UUID=$data_uuid ${mounts[$index]} ext4 defaults,noatime,nofail 0 2"
  line_exists "$swap_line" || printf '%s\n' "$swap_line" >> /etc/fstab
  line_exists "$data_line" || printf '%s\n' "$data_line" >> /etc/fstab
done

systemctl daemon-reload
mount "${mounts[0]}"
mount "${mounts[1]}"
swapon --priority 100 "${devices[0]}p1"
swapon --priority 100 "${devices[1]}p1"

printf '%s\n' 'vm.swappiness=100' 'vm.page-cluster=0' > /etc/sysctl.d/99-deepseek-vllm-swap.conf
sysctl --system >/dev/null

owner_uid="${SUDO_UID:-0}"
owner_gid="${SUDO_GID:-0}"
chown "$owner_uid:$owner_gid" "${mounts[0]}" "${mounts[1]}"
mkdir -p "${mounts[0]}/moet-planes-0731" "${mounts[1]}/moet-store-0731"
chown -R "$owner_uid:$owner_gid" \
  "${mounts[0]}/moet-planes-0731" "${mounts[1]}/moet-store-0731"

printf '%s\n' "NVMe setup complete."
swapon --show
df -h "${mounts[0]}" "${mounts[1]}"
