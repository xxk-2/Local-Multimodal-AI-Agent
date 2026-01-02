import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from tqdm import tqdm
import utils

class LocalAgent:
    def __init__(self):
        print("正在初始化 Local AI Agent...")
        
        # 1. 初始化向量数据库 (持久化存储)
        self.chroma_client = chromadb.PersistentClient(path="./db")
        
        # 创建或获取集合
        # 文本集合 (使用余弦相似度)
        self.text_collection = self.chroma_client.get_or_create_collection(
            name="paper_collection", 
            metadata={"hnsw:space": "cosine"}
        )
        # 图像集合
        self.image_collection = self.chroma_client.get_or_create_collection(
            name="image_collection",
            metadata={"hnsw:space": "cosine"}
        )

        # 2. 加载模型
        # 文本模型: 轻量级，用于论文搜索和分类
        print("加载文本模型 (all-MiniLM-L6-v2)...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 图像模型: CLIP，用于以文搜图
        print("加载多模态模型 (clip-ViT-B-32)...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        
        print("初始化完成！\n")

    def add_and_classify_paper(self, file_path, topics_str):
        """
        处理单篇论文：提取文本 -> 生成嵌入 -> 存入DB -> 语义分类 -> 移动文件
        """
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在。")
            return

        print(f"正在处理: {file_path}")
        
        # A. 提取文本
        content = utils.extract_text_from_pdf(file_path)
        if not content:
            print("无法提取文本，跳过。")
            return

        # B. 生成嵌入并存入数据库
        embedding = self.text_model.encode(content).tolist()
        filename = os.path.basename(file_path)
        
        self.text_collection.upsert(
            documents=[content[:1000]], # 只存前1000字符作为元数据预览
            embeddings=[embedding],
            metadatas=[{"filename": filename, "path": file_path}],
            ids=[filename]
        )
        print(f"索引已更新: {filename}")

        # C. 自动分类 (基于语义相似度)
        if topics_str:
            topics = [t.strip() for t in topics_str.split(',')]
            # 编码论文摘要
            doc_emb = self.text_model.encode(content[:500], convert_to_tensor=True)
            # 编码所有主题
            topic_embs = self.text_model.encode(topics, convert_to_tensor=True)
            
            # 计算相似度
            scores = util.cos_sim(doc_emb, topic_embs)[0]
            
            # 找到最高分的索引
            best_idx = scores.argmax().item()
            
            # 获取对应的主题名称
            best_topic = topics[best_idx]
            
            # [修正点] 使用数字索引 best_idx 获取分数，而不是字符串 best_topic
            score = scores[best_idx].item() 

            print(f"分类结果: '{best_topic}' (置信度: {score:.4f})")
            
            # D. 移动文件
            new_dir = os.path.join(os.path.dirname(file_path), best_topic)
            new_path = utils.move_file(file_path, new_dir)
            
            # 更新数据库中的路径信息
            self.text_collection.update(
                ids=[filename],
                metadatas=[{"filename": filename, "path": new_path}]
            )
            print(f"文件已移动至: {new_path}")

    def search_papers(self, query, n_results=3):
        """
        语义搜索论文
        """
        print(f"正在搜索: '{query}' ...")
        query_emb = self.text_model.encode(query).tolist()
        
        results = self.text_collection.query(
            query_embeddings=[query_emb],
            n_results=n_results
        )
        
        print(f"\n找到 {len(results['ids'][0])} 个相关结果:")
        for i in range(len(results['ids'][0])):
            filename = results['metadatas'][0][i]['filename']
            path = results['metadatas'][0][i]['path']
            distance = results['distances'][0][i]
            # 距离越小越相似 (Cosine Distance)
            print(f"[{i+1}] {filename} (相关性: {1-distance:.4f})")
            print(f"    路径: {path}")

    def index_images(self, image_folder):
        """
        批量索引文件夹中的图片
        """
        if not os.path.exists(image_folder):
            print(f"文件夹不存在: {image_folder}")
            return

        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"发现 {len(image_files)} 张图片，开始索引...")

        for img_file in tqdm(image_files):
            img_path = os.path.join(image_folder, img_file)
            try:
                # 打开图片
                image = Image.open(img_path)
                # CLIP 编码
                embedding = self.clip_model.encode(image).tolist()
                
                self.image_collection.upsert(
                    ids=[img_file],
                    embeddings=[embedding],
                    metadatas=[{"path": img_path}]
                )
            except Exception as e:
                print(f"处理图片 {img_file} 失败: {e}")
        
        print("图片索引完成！")

    def search_images(self, query, n_results=3):
        """
        以文搜图
        """
        print(f"正在搜索图片: '{query}' ...")
        # CLIP 文本编码
        query_emb = self.clip_model.encode(query).tolist()
        
        results = self.image_collection.query(
            query_embeddings=[query_emb],
            n_results=n_results
        )
        
        print(f"\n找到最匹配的图片:")
        for i in range(len(results['ids'][0])):
            img_name = results['ids'][0][i]
            path = results['metadatas'][0][i]['path']
            distance = results['distances'][0][i]
            print(f"[{i+1}] {img_name} (匹配度: {1-distance:.4f})")
            # 在这里可以添加打开图片的代码，例如 os.startfile(path) (Windows)