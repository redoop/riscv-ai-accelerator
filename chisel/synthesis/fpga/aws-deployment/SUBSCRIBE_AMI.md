# 订阅 Vivado AMI 指南

## 📋 订阅步骤

### 1. 访问 AWS Marketplace
打开以下链接：
```
https://aws.amazon.com/marketplace/pp?sku=20lw4py5e675o8vixf1icrpir
```

### 2. 登录 AWS 账户
使用你的 AWS 账户登录

### 3. 点击 "Continue to Subscribe"
在产品页面上找到并点击订阅按钮

### 4. 接受条款
- 阅读并接受软件使用条款
- 点击 "Accept Terms"

### 5. 等待订阅激活
- 通常需要几分钟
- 页面会显示 "Thank you for subscribing"

### 6. 返回终端重新启动
订阅完成后，运行：
```bash
cd chisel/synthesis/fpga/aws-deployment
./launch_f2_vivado.sh
```

## 📌 重要信息

**AMI 详情**：
- AMI ID: `ami-0b359c50bdba2aac0`
- 名称: FPGA Developer AMI
- 预装: Vivado 2025.1
- 区域: us-east-1

**费用说明**：
- AMI 本身通常免费（仅收取 EC2 实例费用）
- F2.6xlarge Spot 价格: ~$1.00/小时
- 按实际使用时间计费

## ✅ 订阅完成后

运行启动脚本：
```bash
./launch_f2_vivado.sh
```

实例启动后会显示：
- 实例 ID
- 公网 IP
- SSH 连接命令
- Vivado 验证命令

## 🔍 验证 Vivado

连接到实例后验证：
```bash
vivado -version
```

应该看到：
```
Vivado v2025.1 (64-bit)
```

## 📞 遇到问题？

如果订阅后仍然失败：
1. 等待 5-10 分钟让订阅生效
2. 检查 AWS 账户权限
3. 确认区域是 us-east-1
4. 查看 AWS Marketplace 订阅状态
