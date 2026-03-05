
- [PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate)
- [PDFMathTranslate-next](https://github.com/PDFMathTranslate-next/PDFMathTranslate-next)

如果只是个人使用，python版本的可能更简单方便。下面是docker版本的部署方案。

## 一、准备环境

### 1. 安装 Docker Desktop

- [docker-desktop](https://www.docker.com/products/docker-desktop/)

安装完成后确认：

```bash
docker --version
```

### 2. ollama

- [ollama](https://ollama.com)

### 3. 拉取镜像

```bash
docker pull byaidu/pdf2zh

# 2.0 版本（next）
docker pull awwaawwa/pdfmathtranslate-next
```

## 二、创建数据目录

建两个目录(可以自定义，但是要跟docker), windows 类似如下：

```
D:\pdf2zh\output
D:\pdf2zh\cache
```


### 三、启动容器

```bash
# 严格模式启动
docker run -d --name pdf2zh -p 7860:7860 -v D:\pdf2zh\output:/app/pdf2zh_files -v D:\pdf2zh\cache:/root/.cache --restart=always byaidu/pdf2zh

# 兼容模式启动
docker run -d --name pdf2zh -p 7860:7860 -v D:\pdf2zh\output:/app/pdf2zh_files -v D:\pdf2zh\cache:/root/.cache --restart=always byaidu/pdf2zh pdf2zh -i --compatible
```

参数说明：
```
-p 7860:7860                    Web界面端口
-v output:/app/pdf2zh_files     保存翻译结果
-v cache:/root/.cache           保存模型
--restart                       Docker重启自动启动
pdf2zh -i --compatible          兼容模式启动（如果要放弃有格式问题的翻译，则不需要这个）
```

## 四、问题排查

### 查看容器

```bash
docker ps
```

必须看到：
```
0.0.0.0:7860->7860/tcp
```

### 打开网页

本机访问
```
http://localhost:7860
```

局域网访问
```
http://部署电脑IP:7860
```

### 端口权限

局域网访问失败, 检查 Windows 防火墙是否允许 7860 端口。如果没有，下面是windows系统开放对应窗口权限的指令。
```powershell
New-NetFirewallRule -DisplayName 'pdfzh2' -Profile @('Domain', 'Public', 'Private') -Direction Inbound -Action Allow -Protocol TCP -LocalPort 7860

New-NetFirewallRule -DisplayName 'ollama' -Profile @('Domain', 'Public', 'Private') -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
```

### 常用管理命令

有很多问题可以通过重启解决。
```bash
# 查看日志
docker logs -f pdf2zh

# 停止容器
docker stop pdf2zh

# 开启容器 
docker start pdf2zh

# 删除容器
docker rm -f pdf2zh
```

### 查看容器内文档

```bash
docker exec -it pdf2zh bash
find / -name "*dual.pdf" 2>/dev/null
```
## 五、使用方法

1. 打开网页
2. 上传 PDF
3. 检查配置:
```
Ollama
http://192.168.50.50:11434
gemma3:27b-it-qat
```

4. 开始翻译

翻译完成后的文档网页上也可以直接下载，文件会保存到(之间容器映射的文件夹地址)：
```
D:\pdf2zh\output
```
即使关闭网页也不会丢。

### 文档条件

文档需要是矢量PDF，不能是扫描版PDF。扫描版PDF可以用MineU之类的软件转换矢量PDF后再翻译。

### 翻译速度

翻译速度主要LLM速度有关，使用4090 &`gemma3:27b-it-qat` 模型 80tokens/s 的前提下大约 10-20s/页）。

## 六、更新版本

```bash
docker rm -f pdf2zh
docker pull byaidu/pdf2zh
docker run -d --name pdf2zh -p 7860:7860 -v D:\pdf2zh\output:/app/pdf2zh_files -v D:\pdf2zh\cache:/root/.cache --restart=always byaidu/pdf2zh pdf2zh -i --compatible
```
