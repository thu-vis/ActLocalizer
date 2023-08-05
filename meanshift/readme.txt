库生成：执行以下指令，可以生成对应系统和环境下DAGSpace库
- cd DAGSpace/src
- python setup.py build_ext --inplace

库使用：
- 导入：from DAGSpace import Space
- 初始化：可以使用距离矩阵（整数）进行初始化
- 聚类：可以使用 Space.MeanShift/...等函数进行聚类

项目后端聚类使用：
将库添加到 application\views\model_utils
修改 application\views\model_utils\cluster_helper.py 中 type="DAGSpace"