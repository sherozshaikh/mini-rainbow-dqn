# Deploy to AWS Free-Tier VM

Step-by-step guide to deploy Mini-Rainbow DQN on an AWS `t2.micro` instance (free tier).

---

## Prerequisites

- AWS account (free tier eligible for 12 months)
- AWS CLI installed locally (optional, you can use the AWS Console)

---

## 1. Create the EC2 Instance

### Via Console

1. Go to [EC2 > Launch Instance](https://console.aws.amazon.com/ec2/v2/home#LaunchInstances)
2. Configure:
   - **Name:** `mini-rainbow-dqn`
   - **AMI:** Ubuntu Server 22.04 LTS (free tier eligible)
   - **Instance type:** `t2.micro` (free tier: 1 vCPU, 1 GB RAM)
   - **Key pair:** Create or select an existing key pair
   - **Network settings:** Allow SSH (port 22) and Custom TCP (port 8000) from `0.0.0.0/0`
   - **Storage:** 30 GB gp2 (free tier allows up to 30 GB)
3. Click **Launch Instance**

### Via CLI

```bash
# Create a security group
aws ec2 create-security-group \
    --group-name mini-rainbow-sg \
    --description "Mini-Rainbow DQN platform"

# Allow SSH and port 8000
aws ec2 authorize-security-group-ingress \
    --group-name mini-rainbow-sg \
    --protocol tcp --port 22 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-name mini-rainbow-sg \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0

# Launch instance
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type t2.micro \
    --key-name your-key-pair \
    --security-groups mini-rainbow-sg \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mini-rainbow-dqn}]'
```

Note: Replace `ami-0c7217cdde317cfec` with the latest Ubuntu 22.04 AMI for your region. Find it at [Ubuntu AMI Locator](https://cloud-images.ubuntu.com/locator/ec2/).

---

## 2. SSH into the Instance

```bash
ssh -i your-key-pair.pem ubuntu@<PUBLIC_IP>
```

Find the public IP in the EC2 Console under your instance details.

---

## 3. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
```

Log out and back in:

```bash
exit
ssh -i your-key-pair.pem ubuntu@<PUBLIC_IP>
```

---

## 4. Add Swap (recommended for t2.micro)

The t2.micro has 1 GB RAM. Adding swap prevents OOM kills:

```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 5. Pull and Run

```bash
sudo docker pull sherozshaikh/mini-rainbow-dqn:v1.0.0
sudo docker run -d \
    --name mini-rainbow \
    --restart unless-stopped \
    -p 8000:8000 \
    sherozshaikh/mini-rainbow-dqn:v1.0.0
```

---

## 6. Access the Platform

Open `http://<PUBLIC_IP>:8000` in your browser.

Find your instance's public IP:

```bash
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=mini-rainbow-dqn" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text
```

---

## 7. (Optional) Assign an Elastic IP

By default, the public IP changes when you stop/start the instance:

### Via Console

1. Go to [EC2 > Elastic IPs](https://console.aws.amazon.com/ec2/v2/home#Addresses)
2. Click **Allocate Elastic IP address** > **Allocate**
3. Select the new IP > **Actions** > **Associate Elastic IP address**
4. Select your `mini-rainbow-dqn` instance > **Associate**

### Via CLI

```bash
ALLOC_ID=$(aws ec2 allocate-address --query 'AllocationId' --output text)
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=mini-rainbow-dqn" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text)
aws ec2 associate-address --allocation-id $ALLOC_ID --instance-id $INSTANCE_ID
```

Note: Elastic IPs are free when associated with a running instance. You are charged when the IP is allocated but not associated.

---

## Start / Stop the Instance

To avoid charges when not using the instance:

```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=mini-rainbow-dqn" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text)

# Stop
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Start
aws ec2 start-instances --instance-ids $INSTANCE_ID
```

The Docker container has `--restart unless-stopped`, so it starts automatically when the instance boots.

---

## Verify

```bash
curl http://<PUBLIC_IP>:8000/health
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
| Can't access port 8000 | Check security group allows TCP 8000: EC2 > Security Groups |
| Connection timeout | Verify instance is running and public IP is correct |
| Container exits immediately | Check logs: `sudo docker logs mini-rainbow` |
| Out of memory | Verify swap is active: `free -h` |
| Slow first load | PyTorch model loading takes ~10s on first request |
| Public IP changed after restart | Assign an Elastic IP (step 7) |
| Free tier expiring | t2.micro is free for 12 months from account creation |
