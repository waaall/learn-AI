
## VLLM 部署 VLM


- [Qwen3-VL-30B-AWQ-4bit量化模型](https://huggingface.co/cyankiwi/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit)

### 模型下载

```bash
# 下载 huggingface_hub
pip install huggingface_hub

# 可选 HF-mirror
hf download cyankiwi/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit --local-dir D:/dev_software/AI_models/huggingface/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit

# 如果无法下载可能需要登陆和授权问题
hf auth login
```

### 模型部署


docker 部署具体见
`AI-config/vllm-docker-compose/compose-qwen3vl-4090_1.yml`


### 模型调用示例

```bash
# Call the server using curl:
curl -X POST "http://localhost:8124/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "Qwen3-VL-30B-A3B-Instruct-AWQ-4bit",
		"messages": [
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": "Describe this image in one sentence."
					},
					{
						"type": "image_url",
						"image_url": {
							"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
						}
					}
				]
			}
		]
	}'
```