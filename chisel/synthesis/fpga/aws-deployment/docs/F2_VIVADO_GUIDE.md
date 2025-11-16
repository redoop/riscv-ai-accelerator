# F2 实例 Vivado 使用指南

## ✅ 实例信息
- **IP**: 54.81.161.62
- **用户名**: ubuntu
- **实例 ID**: i-00d976d528e721c43
- **Vivado 版本**: 2025.1
- **Vivado 路径**: `/tools/Xilinx/2025.1/Vivado/bin/vivado`

## 🔌 连接实例
```bash
ssh -i ~/.ssh/fpga-f2-key.pem ubuntu@54.81.161.62
```

## 🛠️ 设置 Vivado 环境

### 方法 1: 直接使用完整路径
```bash
/tools/Xilinx/2025.1/Vivado/bin/vivado -version
```

### 方法 2: 添加到 PATH
```bash
export PATH="/tools/Xilinx/2025.1/Vivado/bin:$PATH"
vivado -version
```

### 方法 3: 使用设置脚本（推荐）
```bash
# 上传设置脚本
scp -i ~/.ssh/fpga-f2-key.pem setup_vivado_env.sh ubuntu@54.81.161.62:~/

# 在实例上执行
ssh -i ~/.ssh/fpga-f2-key.pem ubuntu@54.81.161.62
source ~/setup_vivado_env.sh
```

## 📤 上传项目文件

### 上传单个文件
```bash
scp -i ~/.ssh/fpga-f2-key.pem your_design.v ubuntu@54.81.161.62:~/
```

### 上传整个目录
```bash
scp -i ~/.ssh/fpga-f2-key.pem -r your_project/ ubuntu@54.81.161.62:~/
```

### 上传压缩包
```bash
# 本地打包
tar czf project.tar.gz your_project/

# 上传
scp -i ~/.ssh/fpga-f2-key.pem project.tar.gz ubuntu@54.81.161.62:~/

# 在实例上解压
ssh -i ~/.ssh/fpga-f2-key.pem ubuntu@54.81.161.62 'tar xzf project.tar.gz'
```

## 🚀 运行 Vivado

### 批处理模式（推荐用于综合）
```bash
vivado -mode batch -source synthesis.tcl
```

### TCL 模式
```bash
vivado -mode tcl
```

### GUI 模式（需要 X11 转发）
```bash
# 本地连接时启用 X11 转发
ssh -i ~/.ssh/fpga-f2-key.pem -X ubuntu@54.81.161.62
vivado
```

## 📝 示例 TCL 综合脚本

创建 `synthesis.tcl`:
```tcl
# 创建项目
create_project my_project ./project_dir -part xcu280-fsvh2892-2L-e

# 添加源文件
add_files {design.v}
add_files -fileset constrs_1 {constraints.xdc}

# 运行综合
synth_design -top top_module

# 生成报告
report_timing_summary -file timing_summary.rpt
report_utilization -file utilization.rpt

# 保存检查点
write_checkpoint -force post_synth.dcp

# 退出
exit
```

运行：
```bash
vivado -mode batch -source synthesis.tcl
```

## 📊 查看资源使用

### 磁盘空间
```bash
df -h
```

### 内存使用
```bash
free -h
```

### CPU 信息
```bash
lscpu
```

## 📥 下载结果

### 下载单个文件
```bash
scp -i ~/.ssh/fpga-f2-key.pem ubuntu@54.81.161.62:~/result.bit ./
```

### 下载整个目录
```bash
scp -i ~/.ssh/fpga-f2-key.pem -r ubuntu@54.81.161.62:~/project_dir ./
```

### 下载并压缩
```bash
# 在实例上压缩
ssh -i ~/.ssh/fpga-f2-key.pem ubuntu@54.81.161.62 'tar czf results.tar.gz project_dir/'

# 下载
scp -i ~/.ssh/fpga-f2-key.pem ubuntu@54.81.161.62:~/results.tar.gz ./
```

## 🛑 停止实例

### 使用 AWS CLI
```bash
aws ec2 terminate-instances --instance-ids i-00d976d528e721c43 --region us-east-1
```

### 检查实例状态
```bash
aws ec2 describe-instances --instance-ids i-00d976d528e721c43 --region us-east-1 --query 'Reservations[0].Instances[0].State.Name'
```

## 💡 最佳实践

1. **定期保存结果**: 将重要文件下载到本地
2. **使用批处理模式**: 避免 GUI 开销
3. **监控资源**: 确保不超出实例限制
4. **及时停止**: 用完立即终止实例节省费用
5. **使用 tmux/screen**: 长时间运行的任务使用会话管理

## 🔧 故障排查

### Vivado 找不到
```bash
find /tools -name vivado -type f 2>/dev/null
```

### 许可证问题
```bash
echo $LM_LICENSE_FILE
```

### 内存不足
```bash
# 检查可用内存
free -h

# 清理缓存
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

## 📞 快速参考

| 操作 | 命令 |
|------|------|
| 连接实例 | `ssh -i ~/.ssh/fpga-f2-key.pem ubuntu@54.81.161.62` |
| Vivado 版本 | `/tools/Xilinx/2025.1/Vivado/bin/vivado -version` |
| 上传文件 | `scp -i ~/.ssh/fpga-f2-key.pem file ubuntu@54.81.161.62:~/` |
| 下载文件 | `scp -i ~/.ssh/fpga-f2-key.pem ubuntu@54.81.161.62:~/file ./` |
| 停止实例 | `aws ec2 terminate-instances --instance-ids i-00d976d528e721c43` |
