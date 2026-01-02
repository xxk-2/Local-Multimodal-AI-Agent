import argparse
from agent import LocalAgent

def main():
    parser = argparse.ArgumentParser(description="本地多模态 AI 智能助手")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 命令 1: 添加/分类论文
    parser_add = subparsers.add_parser("add_paper", help="添加并分类 PDF 论文")
    parser_add.add_argument("path", type=str, help="PDF 文件路径")
    parser_add.add_argument("--topics", type=str, default="Uncategorized", help="分类主题，用逗号分隔 (例如: 'CV,NLP,RL')")

    # 命令 2: 搜索论文
    parser_search = subparsers.add_parser("search_paper", help="语义搜索论文")
    parser_search.add_argument("query", type=str, help="搜索关键词或问题")

    # 命令 3: 索引图片目录
    parser_idx_img = subparsers.add_parser("index_images", help="索引指定目录下的所有图片")
    parser_idx_img.add_argument("folder", type=str, help="图片文件夹路径")

    # 命令 4: 以文搜图
    parser_search_img = subparsers.add_parser("search_image", help="通过自然语言描述搜索图片")
    parser_search_img.add_argument("query", type=str, help="图片描述 (例如: 'a dog on the grass')")

    args = parser.parse_args()

    if args.command:
        # 懒加载：只有在有命令时才初始化模型，加快 help 显示速度
        agent = LocalAgent()
        
        if args.command == "add_paper":
            agent.add_and_classify_paper(args.path, args.topics)
        elif args.command == "search_paper":
            agent.search_papers(args.query)
        elif args.command == "index_images":
            agent.index_images(args.folder)
        elif args.command == "search_image":
            agent.search_images(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()