---
title: 【ROS2】#1 机器人操作系统简介与节点编写
date: 2026-03-15
summary: ROS2是用于构建机器人应用程序的一套软件库和工具集。
tags: [ROS2, 节点, rclcpp, rclpy]
category: ROS2
---

**阅读须知**

本系列由博主在学习@鱼香ROS的《动手学ROS2》过程中对其重新编排而成，尝试在保证可学性的同时力求极致的简明扼要。

代码将基于学习进度考量，被解耦为**框架**和**自定义内容**进行解释，旨在减轻理解压力，但至少需要一定的C++和Python基础。考虑到学习需求的复杂性，博主还是会在文末附录结合自身情况补充一些细节详解。

- **框架**：固定不变的代码，仅以行为单位，在代码块中以注释的语法适当解释代码功能，不深入函数、参数、语法等细节。
- **自定义内容**：博主假定读者有能力自由调整的代码片段，在代码块内会以序号`[x]`在注释中标记所在行，在代码块后解释其作用和范围。

已经出现过的框架和自定义内容不会重复解释，仅关注代码中新出现的，或需要对既有解释进行拓展的部分。

呈现终端命令的代码块博主将在开头进行注释：

```bash
# ~/file1/file2 A
```

表示终端在自定义文件夹（`~`）下的子目录`file1/file2`中打开，并命名为终端`A`，后续在现有终端中继续运行命令则只注释终端名`A`。无需重复使用的终端不进行命名。

介绍新命令时，命令中的参数和自定义名称使用`<>`标记，实际输入无需带`<>`。但是在非终端运行的背景下`<>`会引起歧义，需留意注释，通常以`xxxx_name`形式呈现的内容均为自定义名称。为了便于实操，正式使用终端命令时会用不带`<>`的明确的参数和自定义名称。

出于平台编辑功能限制，文章发布后的勘误、补充和调整工作均只在博主的个人网站www.inkem.space上进行，因此推荐移步阅读。

# ROS2

**ROS2**（Robot Operating System 2，机器人操作系统2）是一个用于构建机器人应用程序的一套软件库和工具集。在复杂的机器人系统中，会有很多不同的部件同时运行，例如传感器数据处理、路径规划、执行器控制等。ROS2的主要作用是让这些分散的部件能够方便、高效、可靠地相互通信，让编写模块化、可复用的机器人软件变得更容易。

通过以下概念我们可以初步了解ROS2项目的基本框架：

- **工作空间**（Workspace）：存放所有与机器人项目相关内容的顶级目录；
- **功能包**（Package）：组织代码的最小独立功能单元，封装实现同一核心功能的资源，方便复用；
- **节点**（Node）：功能包内部的可执行程序，是实现功能包功能的最小通信单元；
- **通信接口**（Interface）：节点间进行数据交换的方式，包括话题、服务、动作和参数四种类型（后续篇章介绍）。

![|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260314201457.png)

---

**一键安装ROS2**

本节使用 @鱼香ROS 编写的一键安装ROS2脚本程序，可以非常便利地安装ROS2。

安装环境：Ubuntu22.04操作系统

```bash
# ~

wget http://fishros.com/install -O fishros && . fishros

# 选择1-一键安装ROS
# 选择更换系统源
# 选择humble版本ROS2
# 选择桌面版
```

---

编写ROS2项目的第一步是创建工作空间和功能包。

**创建工作空间**

工作空间的结构包括：

- `src`：源代码空间，存放所有需要编译的源代码；
- `build`：编译空间，存放编译过程的中间文件和缓存文件；
- `install`：安装空间，存放最终编译产物（可执行文件、库文件、配置文件）；
- `log`：日志空间，存放运行时日志和编译日志。

只有`src`目录需要手动构建并加入Git版本控制。

```bash
# ~ A

# 创建多级目录<workspace_name>/src
mkdir -p <workspace_name>/src
```

- `workspace_name`：工作空间的名称。

**创建功能包**

```bash
# A

# 进入目录<workspace_name>/src
cd <workspace_name>/src
# 创建功能包<package_name>
ros2 pkg create <package_name> --build-type <build_type> --dependencies <dependencies>
```

- `package_name`：功能包的名称；
  - 只能包含小写字母、数字和下划线，并以字母开头。
- `build_type`：指定功能包的编译类型：
  - `ament_cmake`：用于C++编写的功能包；
  - `ament_python`：用于Python编写的功能包；
  - `cmake`：CMake项目，不常用。
- `dependencies`：指定功能包的依赖，此处先给出在C++和Python中使用ROS2最基础的依赖：
  - `rclcpp`：ROS2的C++客户端接口；
  - `rclpy`：ROS2的Python客户端接口。

```bash
# A

# 创建C++功能包
ros2 pkg create <package_name> --build-type ament_cmake --dependencies rclcpp
# 创建Python功能包
ros2 pkg create <package_name> --build-type ament_python --dependencies rclpy
```

**C++功能包自动生成的文件结构**

- `package_name/`
  - `CMakeLists.txt`：CMake构建系统的配置文件；
  - `include/package_name/`：头文件目录，存放公共头文件（.hpp或.h）；
  - `package.xml`：功能包的清单文件；
  - `src/`：源代码目录，存放源代码文件（.cpp）。

**Python功能包自动生成的文件结构**

- `package_name/`
  - `package_name/`：代码目录，存放节点代码：
    - `__init__.py`：标志文件，告诉Python编辑器这是一个包。
  - `resource/`：资源目录，存放功能包的标志文件：
    - `package_name`：标志文件，告诉构建工具这个包提供了名为`package_name`的资源。
  - `test/`：测试目录，存放单元测试和代码风格检查脚本：
    - `test_copyright.py`：检查所有Python文件头部是否包含正确的版权声明；
    - `test_flake8.py`：检查代码是否符合PEP8风格指南；
    - `test_pep257.py`：检查文档字符串是否符合PEP257规范。
  - `package.xml`：功能包的清单文件；
  - `setup.py`：Python包的构建脚本；
  - `setup.cfg`：配置`setup.py`的行为。

_ROS2也可以创建同时包含C++和Python代码的混合功能包，但其结构较复杂，将考虑在后续篇章介绍。_

# 节点

**创建工作空间**

```bash
# ~ B
mkdir -p workspace_name/src
```

## 使用RCLCPP编写节点

**创建功能包**

```bash
# B

cd workspace_name/src
ros2 pkg create cpp_package_name --build-type ament_cmake --dependencies rclcpp
```

在`cpp_package_name/src/`下创建`cpp_node_name.cpp`文件。

**node_name.cpp**

一个最小节点的C++代码如下：

```cpp
// 包含头文件
#include "rclcpp/rclcpp.hpp"

int main(int argc, char **argv)
{
	// 初始化
    rclcpp::init(argc, argv);
    // [1]创建名为cpp_node_name的节点
    auto node = std::make_shared<rclcpp::Node>("cpp_node_name");
    // [2]打印信息“cpp_node_name节点已经启动。”
    RCLCPP_INFO(node->get_logger(), "cpp_node_name节点已经启动。");
    // 保持节点运行并检测退出信号
    rclcpp::spin(node);
    // 关闭rclpy
    rclcpp::shutdown();
    return 0;
}
```

- `[1]cpp_node_name`：节点名称，节点在ROS网络中的唯一标识。
  - 只能包含字母、数字和下划线；
  - 含`~`和`\`的节点名称涉及命名空间的概念，暂不引入。
- `[2]cpp_node_name节点已经启动。`：启动节点后打印的日志信息。

**CMakeLists.txt**

每个节点源代码`cpp_node_name.cpp`都需要在该文档最后写入：

```cmake
# 创建名为cpp_node_name的可执行文件，该可执行文件由源代码文件src/cpp_node_name.cpp编译链接而成
add_executable(cpp_node_name src/cpp_node_name.cpp)
# 为可执行文件cpp_node_name添加依赖项rclcpp
ament_target_dependencies(cpp_node_name rclcpp)

# 将编译好的可执行文件cpp_node_name复制到相对路径${PROJECT_NAME}下
install(TARGETS
cpp_node_name
DESTINATION lib/${PROJECT_NAME}
)
```

**编译运行节点**

```bash
# ~/workspace_name

# 使用colcon构建工具编译节点
colcon build
# 激活ROS2工作空间的环境
source install/setup.bash
# 运行功能包cpp_package_name下的节点cpp_node_name
ros2 run cpp_package_name cpp_node_name
```

运行结果：

```bash
[INFO] [1773567866.162894123] [cpp_node_name]: cpp_node_name节点已经启动。
```

按`Ctrl+C`退出。

**colcon构建工具**

colcon是一个功能包构建工具，用于编译工作空间下的功能包。其安装命令为：

```bash
# ~
sudo apt-get install python3-colcon-common-extensions
```

现在可以学习的colcon命令：

```bash
# ~/<workspace_name>

# 编译工作空间下的所有功能包
colcon build
# 只编译功能包package_name
colcon build --packages-select <package_name>
```

**测试节点**

在节点运行时，可以运行以下命令测试节点：

```bash
# ~

# 查看节点列表
ros2 node list
# 查看节点node_name的信息
ros2 node info <node_name>
```

---

使用**面向对象编程**（Object Oriented Programming，OOP）编写ROS2节点可以更好地对代码进行维护、扩展和测试。

**cpp_node_name.cpp**

以下是OOP版本的最小节点的C++代码：

```cpp
#include "rclcpp/rclcpp.hpp"

// [1]创建一个名为Node_Name的自定义节点类，继承自rclcpp::Node，表示这是一个ROS2节点
class Node_Name : public rclcpp::Node
{

public:
	// 调用父类构造函数接收一个字符串作为节点名称
    Node_Name(std::string name) : Node(name)
    {
    	//[2]创建节点时打印"大家好，我是<节点名称>。"
        RCLCPP_INFO(this->get_logger(), "大家好，我是%s。", name.c_str());
    }

private:

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    //[3]创建名称为cpp_node_name的Node_Name类节点实例
    auto node = std::make_shared<Node_Name>("cpp_node_name");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
```

- `[1]Node_Name`：节点类名称；
- `[2]大家好，我是%s。`：创建节点时打印的日志信息；
- `[3]cpp_node_name`：节点实例名称。

修改源代码文件后需要重新编译再运行。

## 使用RCLPY编写节点

**创建功能包**

```bash
# B
ros2 pkg create py_package_name --build-type ament_python --dependencies rclpy
```

在`package_name/py_package_name`下创建`py_node_name.py`文件。

**py_node_name.py**

一个最小节点的Python代码如下：

```python
# 导入库文件
import rclpy
from rclpy.node import Node

def main(args=None):
	# 初始化rclpy
    rclpy.init(args=args)
    # [1]创建名为py_node_name的节点
    node = Node("py_node_name")
    # [2]打印信息“py_node_name节点已经启动。”
    node.get_logger().info("py_node_name节点已经启动。")
    # 保持节点运行并检测退出信号
    rclpy.spin(node)
    # 关闭rclpy
    rclpy.shutdown()
```

- `[1]py_node_name`：节点名称。
- `[2]py_node_name节点已经启动。`：启动节点后打印的日志信息。

**setup.py**

每个节点的源代码`py_node_name.py`都需要为`setup()`末尾的`entry_points`参数中的`console_scripts`关键字添加一条映射规则：

```python
entry_points={
    'console_scripts': [
    	# 将py_node_name命令映射到代码目录py_package_name/下的py_node_name.py文件，入口函数为main()
        "py_node_name = py_package_name.py_node_name:main"
    ],
},
```

当`ros2 run`命令的运行对象为`py_node_name`时，Python解释器会执行`py_package_name/py_node_name.py`文件中的`main()`函数。

**编译运行节点**

```bash
# ~/workspace_name
colcon build
source install/setup.bash
ros2 run py_package_name py_node_name
```

运行结果：

```bash
[INFO] [1773568196.391497035] [py_node_name]: py_node_name节点已经启动。
```

---

**py_node_name.py**

以下是OOP版本的最小节点的Python代码：

```python
import rclpy
from rclpy.node import Node

# [1]创建一个名为Node_Name的自定义节点类，继承自Node，表示这是一个ROS2节点
class Node_Name(Node):
    def __init__(self,name):
    	# 调用父类构造函数接收一个字符串作为节点名称
        super().__init__(name)
        # [2]创建节点时打印"大家好，我是<节点名称>。"
        self.get_logger().info("大家好，我是%s!" % name)

def main(args=None):
    rclpy.init(args=args)
    # [3]创建名称为py_node_name的Node_Name类节点实例
    node = Node_Name("py_node_name")
    rclpy.spin(node)
    rclpy.shutdown()
```

- `[1]Node_Name`：节点类名称；
- `[2]大家好，我是%s。`：创建节点时打印的日志信息；
- `[3]py_node_name`：节点实例名称。

# 附录

## rclcpp API

```cpp
void rclcpp::init(
    int argc,
    char const *const *argv,
    const InitOptions &init_options = InitOptions(),
    SignalHandlerOptions signal_handler_options = SignalHandlerOptions::All
);
```

ROS2 C++程序必须首先调用的初始化函数，负责启动ROS2的通信层，并为整个节点进程设置必要的全局状态，包括：

- 通过RMW初始化通信中间件；
- 解析并从`argc`和`argv`中移除命令行中的ROS2本身配置的参数，以便在后续代码解析自己的命令行参数时无需处理ROS2相关参数（节点参数部分详见后续篇章）；
- 创建全局上下文`rclcpp::Context`对象，其封装了节点及其他类似实体之间的共享状态，同时表示rclcpp从初始化到关闭之间的生命周期；
- 安装信号处理器，设置全局的信号处理函数，例如在终端按下`Ctrl+C`时触发`rclcpp::shutdown()`，从而清理资源并让程序正常退出。

参数：

- `argc`和`argv`：来自`main`函数的命令行参数个数和参数字符串数组，被传入以供ROS2解析其特定参数；
- `init_options`：可选参数，允许在初始化时进行更细致的配置；
- `signal_handler_options`：可选参数，用于控制安装哪些信号处理器，默认安装所有信号处理函数。

```cpp
RCLCPP_INFO(
	rclcpp::Logger logger,
	const char* format,
	...
)
```

日志输出宏，将一条信息级别的日志消息打印到终端，并同时发送到ROS2的日志系统，其自动包含时间戳、节点名称、日志级别等信息，并可以被重定向、过滤和记录到文件中，是ROS2调试和状态监控的标准工具。

参数：

- `logger`：指定日志消息属于哪个来源，有助于在过滤日志时区分不同节点的输出，通常使用`node->get_logger()`获取；
- `format`：日志消息的文本格式，支持格式说明符如`%d`、`%f`、`%s`等；
- `...`：可变参数，对应`format`字符串中的占位符，提供要打印的具体数据。
  - 配合`%s`的`std::string`类型需要使用`.c_str()`转换。

所有级别的日志宏：

| 宏名称         | 日志级别 | 使用场景                                                         |
| -------------- | -------- | ---------------------------------------------------------------- |
| `RCLCPP_DEBUG` | DEBUG    | 调试信息，仅在开发调试时使用，生产环境通常关闭。                 |
| `RCLCPP_INFO`  | INFO     | 标准输出，程序正常运行时的提示信息。                             |
| `RCLCPP_WARN`  | WARN     | 警告信息，表示可能存在问题，但不影响程序继续运行。               |
| `RCLCPP_ERROR` | ERROR    | 错误信息，表示发生了错误，可能导致功能异常，但程序可能仍能继续。 |
| `RCLCPP_FATAL` | FATAL    | 致命错误，表示发生了严重错误，程序即将终止。                     |

```cpp
void rclcpp::spin(
    std::shared_ptr<rclcpp::node_interfaces::NodeBaseInterface::SharedPtr> node_ptr
);
```

最核心的时间处理循环，执行以下任务：

- 调用底层中间件的等待函数，阻塞当前线程直到有事件发生；
- 当事件发生时，识别事件触发源并调用相应的回调函数；
- 保持程序运行直到节点被关闭。

参数：

- `node_ptr`：所处理节点的共享指针，更常用的类型是`std::shared_ptr<rclcpp::Node>`，由函数内部自动获取节点的`NodeBaseInterface`。

```cpp
bool rclcpp::shutdown(
	rclcpp::Context::SharedPtr context = nullptr,
	const std::string & reason = "user called rclcpp::shutdown()"
)
```

退出的关键函数，在被调用时关闭上下文，进而通知一切与之相关的ROS2组件停止运行，并清理全局资源。

参数：

- `context`：可选参数，指定要关闭的上下文，若为`nullptr`则关闭全局上下文；
- `reason`：可选参数，关闭原因字符串，用于日志记录说明关闭原因。

返回：

- 如果关闭成功则返回`True`，如果上下文已经关闭则返回`False`。

## rclpy API

`rclpy.函数名`记法：

```python
def 函数名(
	参数名: (Optional[]为可选参数)参数类型 = 参数默认值
) -> 返回值类型:
```

```python
def init(
    args: Optional[List[str]] = None,
    context: Optional[Context] = None
) -> None:
```

ROS2 Python程序必须首先调用的初始化函数，与`rclcpp::init`类似。

参数：

- `args`：命令行参数的字符串列表；
- `context`：指定要初始化的上下文环境，若为`None`则使用默认上下文。

```python
node = Node('node_name')
node.get_logger().info()
```

每个ROS2节点都有自己的日志记录器，其自动关联节点名称。`get_logger()`返回节点的`rclpy.loggin.Logger`实例，日志级别包括`debug()`、`info()`、`warn()`、`error()`、`fatal()`。与RCLCPP日志输出宏类似。

```python
def spin(
	node: Node,
	executor: Optional[Executor] = None
) -> None:
```

节点时间处理循环，与`rclcpp::spin`类似。

参数：

- `node`：要保持在运行状态的节点；
- `executor`：用于处理回调的执行器，若为`None`则使用全局执行器。

```python
def shutdown(
	context: Optional[Context] = None
)
```

上下文关闭函数，与`rclcpp::shutdown`类似。

参数：

- `context`：指定要关闭的上下文。
