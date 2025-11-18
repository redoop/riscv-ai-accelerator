# F2 实例构建流程（实验性）

⚠️ **警告**: F2 实例不支持 AWS AFI 创建！

本目录包含使用 F2 实例进行本地开发和测试的流程。

## 为什么不推荐 F2？

- ❌ **设备不兼容**: xcvu47p 与 AWS AFI 服务不兼容
- ❌ **成本更高**: Spot 价格约 $2.30/小时
- ❌ **不支持 AFI**: 无法创建 AFI
- ❌ **仅用于开发**: 生成的 DCP 无法用于生产

## 适用场景

F2 实例仅适用于以下场景：

1. **本地开发和调试**
   - 不需要创建 AFI
   - 仅用于设计验证

2. **大型设计**
   - 需要 9M LUTs（F1 只有 2.5M）
   - 但无法部署到 AWS

3. **实验性功能**
   - 测试新的 Vivado 特性
   - 不需要云端部署

## 推荐方案

**如果需要创建 AFI，请使用 F1 实例！**

```bash
cd ../f1
bash launch.sh
```

## F2 流程（仅供参考）

### 1. 启动 F2 实例
```bash
cd aws-deployment/f2
bash launch.sh
```

### 2. 上传项目
```bash
bash upload.sh
```

### 3. 启动构建
```bash
bash build.sh
```

### 4. 下载 DCP（仅用于本地分析）
```bash
bash download_dcp.sh
```

⚠️ **注意**: 生成的 DCP 无法用于创建 AFI！

## 成本对比

| 实例 | Spot 价格 | 4小时成本 | AFI 支持 |
|------|-----------|----------|----------|
| F1 | $0.50/hr | $2.00 | ✅ |
| F2 | $2.30/hr | $9.20 | ❌ |

## 技术规格

- **实例类型**: f2.6xlarge
- **FPGA 设备**: xcvu47p (Virtex UltraScale+ VU47P)
- **逻辑单元**: 9M LUTs
- **Vivado 版本**: 2024.1
- **AFI 兼容性**: ❌ 不兼容

## 错误示例

如果尝试使用 F2 DCP 创建 AFI，会遇到：

```
ERROR: [Constraints 18-884] HDPRVerify-01: 
design check point is using device xcvu47p, 
yet AWS Shell is using device xcvu9p.
```

## 总结

**除非有特殊原因，请使用 F1 实例！**

- 更便宜
- 支持 AFI
- 官方推荐
