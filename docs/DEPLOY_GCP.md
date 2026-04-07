# Deploy to GCP Free-Tier VM

Step-by-step guide to deploy Mini-Rainbow DQN on a GCP `e2-micro` instance (free tier).

---

## Prerequisites

- Google Cloud account with billing enabled (free tier eligible)
- `gcloud` CLI installed locally (optional, you can use the GCP Console)

---

## 1. Create the VM

### Via Console

1. Go to [Compute Engine > VM instances](https://console.cloud.google.com/compute/instances)
2. Click **Create Instance**
3. Configure:
   - **Name:** `mini-rainbow-dqn`
   - **Region:** `us-central1` (or any free-tier eligible region: `us-west1`, `us-east1`)
   - **Zone:** any
   - **Machine type:** `e2-micro` (free tier: 1 vCPU, 1 GB RAM)
   - **Boot disk:** Ubuntu 22.04 LTS, 30 GB standard persistent disk
   - **Firewall:** Check both "Allow HTTP traffic" and "Allow HTTPS traffic"
4. Click **Create**

### Via CLI

```bash
gcloud compute instances create mini-rainbow-dqn \
    --zone=us-central1-a \
    --machine-type=e2-micro \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=30GB \
    --tags=http-server
```

---

## 2. Open Port 8000

### Via Console

1. Go to [VPC Network > Firewall](https://console.cloud.google.com/networking/firewalls)
2. Click **Create Firewall Rule**
3. Configure:
   - **Name:** `allow-8000`
   - **Targets:** All instances in the network
   - **Source IP ranges:** `0.0.0.0/0`
   - **Protocols and ports:** TCP, port `8000`
4. Click **Create**

### Via CLI

```bash
gcloud compute firewall-rules create allow-8000 \
    --allow=tcp:8000 \
    --source-ranges=0.0.0.0/0 \
    --description="Allow port 8000 for Mini-Rainbow DQN"
```

---

## 3. SSH into the VM

```bash
gcloud compute ssh mini-rainbow-dqn --zone=us-central1-a
```

Or click **SSH** in the Console.

---

## 4. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
```

Log out and back in for the group change to take effect:

```bash
exit
gcloud compute ssh mini-rainbow-dqn --zone=us-central1-a
```

---

## 5. Add Swap (recommended for e2-micro)

The e2-micro has 1 GB RAM. Adding swap prevents OOM kills:

```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 6. Pull and Run

```bash
sudo docker pull sherozshaikh/mini-rainbow-dqn:v1.0.0
sudo docker run -d \
    --name mini-rainbow \
    --restart unless-stopped \
    -p 8000:8000 \
    sherozshaikh/mini-rainbow-dqn:v1.0.0
```

---

## 7. Access the Platform

Find your VM's external IP:

```bash
gcloud compute instances describe mini-rainbow-dqn \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Open `http://<EXTERNAL_IP>:8000` in your browser.

---

## 8. (Optional) Reserve a Static IP

By default, the external IP changes when you stop/start the VM:

```bash
gcloud compute addresses create mini-rainbow-ip --region=us-central1

gcloud compute instances delete-access-config mini-rainbow-dqn \
    --zone=us-central1-a \
    --access-config-name="External NAT"

gcloud compute instances add-access-config mini-rainbow-dqn \
    --zone=us-central1-a \
    --address=$(gcloud compute addresses describe mini-rainbow-ip --region=us-central1 --format='get(address)')
```

---

## Start / Stop the VM

To avoid charges when not using the VM:

```bash
# Stop (no compute charges while stopped, disk charges still apply)
gcloud compute instances stop mini-rainbow-dqn --zone=us-central1-a

# Start
gcloud compute instances start mini-rainbow-dqn --zone=us-central1-a
```

The Docker container has `--restart unless-stopped`, so it starts automatically when the VM boots.

---

## Verify

```bash
curl http://<EXTERNAL_IP>:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "agents_loaded": ["DQN", "Rainbow-Lite"],
  "total_episodes": {"DQN": 0, "Rainbow-Lite": 0}
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't access port 8000 | Check firewall rule exists: `gcloud compute firewall-rules list` |
| Container exits immediately | Check logs: `sudo docker logs mini-rainbow` |
| Out of memory | Verify swap is active: `free -h` |
| Slow first load | PyTorch model loading takes ~10s on first request |
| External IP changed | Reserve a static IP (step 8) or re-check: `gcloud compute instances list` |
