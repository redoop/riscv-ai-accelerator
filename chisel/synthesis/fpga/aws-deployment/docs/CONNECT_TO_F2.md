# 连接到 F2 实例

**实例信息**：
- 实例 ID: i-0d2843556190e55d4
- 公网 IP: 54.88.54.230
- 用户名: rocky
- 密钥: fpga-dev-key

## 问题：本地没有密钥文件

密钥 `fpga-dev-key` 在 AWS 中存在，但本地没有对应的 .pem 文件。

## 解决方案

### 方案 1：使用现有的 SSH 密钥

如果你有其他 SSH 密钥可以访问实例：

```bash
# 查看可用的密钥
ls -la ~/.ssh/

# 使用默认密钥尝试连接
ssh rocky@54.88.54.230

# 或指定密钥
ssh -i ~/.ssh/id_rsa rocky@54.88.54.230
```

### 方案 2：通过 AWS Session Manager 连接

不需要 SSH 密钥：

```bash
# 安装 Session Manager 插件（如果未安装）
# https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html

# 连接到实例
aws ssm start-session --target i-0d2843556190e55d4 --region us-east-1
```

### 方案 3：创建新密钥对并重启实例

```bash
# 1. 创建新密钥对
aws ec2 create-key-pair \
  --key-name fpga-dev-key-new \
  --region us-east-1 \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/fpga-dev-key-new.pem

chmod 400 ~/.ssh/fpga-dev-key-new.pem

# 2. 停止当前实例
aws ec2 terminate-instances --instance-ids i-0d2843556190e55d4 --region us-east-1

# 3. 使用新密钥启动新实例
# 修改 launch_f2_spot.sh 中的 KEY_NAME="fpga-dev-key-new"
cd chisel/synthesis/fpga/aws-deployment
./launch_f2_spot.sh
```

### 方案 4：使用 AWS 控制台

1. 访问 [EC2 控制台](https://console.aws.amazon.com/ec2/)
2. 找到实例 i-0d2843556190e55d4
3. 点击 "Connect"
4. 选择 "EC2 Instance Connect" 或 "Session Manager"
5. 点击 "Connect" 直接在浏览器中连接

### 方案 5：添加你的公钥到实例

如果你有本地的 SSH 密钥对：

```bash
# 1. 查看你的公钥
cat ~/.ssh/id_rsa.pub

# 2. 使用 AWS Systems Manager 或 EC2 Instance Connect 添加公钥
# 这需要实例支持 EC2 Instance Connect

# 3. 或者通过 user data 在启动时添加
```

## 推荐方案

**最简单**：使用 AWS 控制台的 EC2 Instance Connect

1. 访问 https://console.aws.amazon.com/ec2/
2. 选择实例 i-0d2843556190e55d4
3. 点击 "Connect" 按钮
4. 选择 "EC2 Instance Connect"
5. 用户名输入：rocky
6. 点击 "Connect"

这样可以直接在浏览器中访问实例，无需 SSH 密钥。

## 连接成功后

### 1. 配置 AWS FPGA 环境

```bash
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source sdk_setup.sh
source hdk_setup.sh
```

### 2. 验证环境

```bash
# 检查 Vivado
vivado -version

# 检查 FPGA
lspci | grep Xilinx

# 检查系统资源
nproc  # 应该显示 24
free -h  # 应该显示 ~256GB
```

### 3. 获取项目代码

由于无法直接 SCP，可以：

**选项 A：从 Git 克隆**
```bash
git clone https://github.com/your-repo/riscv-ai-accelerator.git
cd riscv-ai-accelerator/chisel/synthesis/fpga
```

**选项 B：手动创建文件**
```bash
# 创建必要的目录结构
mkdir -p ~/chisel/synthesis/fpga
mkdir -p ~/chisel/generated/simple_edgeaisoc

# 然后手动复制关键文件内容
```

**选项 C：使用 S3**
```bash
# 在本地上传到 S3
aws s3 cp fpga-project.tar.gz s3://your-bucket/

# 在 F2 实例下载
aws s3 cp s3://your-bucket/fpga-project.tar.gz ~/
tar xzf fpga-project.tar.gz
```

## 当前状态

- ✅ F2 实例已启动并运行
- ✅ 实例 IP: 54.88.54.230
- ⏳ 等待连接方式确定
- ⏳ 等待上传项目代码

## 下一步

1. 选择上述连接方案之一
2. 连接到实例
3. 配置 FPGA 环境
4. 获取项目代码
5. 开始 Vivado 构建

---

**提示**：如果你有 GitHub 仓库，最简单的方式是直接在 F2 实例上 git clone。
