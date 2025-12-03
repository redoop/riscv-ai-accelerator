#!/bin/bash
# 使用华为云镜像

HUAWEI_MIRROR="swr.cn-north-4.myhuaweicloud.com/ddn-k8s/ghcr.io/the-openroad-project/openlane:ff5509f65b17bfa4068d5336495ab1718987ff69"

echo "=========================================="
echo "配置华为云镜像"
echo "=========================================="

# 1. 拉取华为云镜像
echo "拉取镜像: $HUAWEI_MIRROR"
docker pull "$HUAWEI_MIRROR"

# 2. 重新标记为原始名称
ORIGINAL_IMAGE="ghcr.io/the-openroad-project/openlane:ff5509f65b17bfa4068d5336495ab1718987ff69"
echo "重新标记为: $ORIGINAL_IMAGE"
docker tag "$HUAWEI_MIRROR" "$ORIGINAL_IMAGE"

# 3. 验证
echo ""
echo "验证镜像:"
docker images | grep "openlane"

echo ""
echo "✅ 完成！现在可以运行 OpenLane"
