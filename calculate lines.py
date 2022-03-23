import os


def code_lines(path_):
    """
    检查整个项目的代码行数
    :param path_: 项目根路径
    :return:
    """
    total_length = 0
    for path_dir, dirs, files in os.walk(path_):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(path_dir, file)
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                lines_ = []
                # 消除空白行
                for line in lines:
                    if not line:
                        continue
                    lines_.append(line)
                length = len(lines)
                total_length = total_length + length

    # print(total_length)
    return total_length


if __name__ == '__main__':
    path = "D:\yolov3-pytorch"
    lines = code_lines(path)
    print(lines)

