# 从 `bind_tools` 到 `ToolMessage`：Agent 的 Tool Calling 到底是怎么运行的？

## 1. 前言

在使用 LangChain 开发 Agent 时，经常会看到类似下面的代码：

```python
llm_with_tools = llm.bind_tools(tools)
```

刚开始接触 Agent 时，很容易形成一个模糊的理解：

> 给模型绑定一些 Tool，之后模型需要时就会自动执行这些 Tool。

但实际情况并不是这样。

在 `codebase-agent` 中，Tool Calling 被拆成了非常清楚的两个部分：

> **LLM 负责决定“调用什么”，Python Runtime 负责真正“执行什么”。**

也就是说，模型本身并不会直接读取本地文件、搜索代码或者访问向量数据库。

模型真正做的是：

> 生成一个结构化的 Tool Call 请求。

然后由 Python Runtime 根据这个请求找到对应的工具并执行。

理解这个基本原则之后，`bind_tools`、`AIMessage`、`ToolCall`、`ToolMessage`、`ToolCallRecord` 以及 Agent Loop 就可以串成一条完整链路。

---

# 2. Tool 是如何被加入 Agent 的？

项目首先会创建 Agent 可以使用的工具，例如：

```text
search_code
read_file
retrieve_context
```

这些 Tool 最终会组成一个工具列表：

```python
tools = [...]
```

之后，`CodebaseAgent` 中会做两件不同的事情：

```python
self.tools = {
    tool.name: tool
    for tool in tools
}

self.llm_with_tools = llm.bind_tools(tools)
```

虽然两行代码都和 Tool 有关，但职责完全不同。

可以理解成：

```text
                 tools
                   │
           ┌───────┴────────┐
           │                │
           ▼                ▼
    llm.bind_tools()     self.tools
           │                │
           ▼                │
    告诉模型有哪些工具       │
           │                │
           ▼                │
          LLM               │
           │                │
           │ Tool Call      │
           └──────────────► Runtime
                            │
                            ▼
                       tool.invoke()
```

其中：

```python
self.llm_with_tools
```

主要服务于 **LLM 决策**。

而：

```python
self.tools
```

主要服务于 **Python Runtime 执行**。

---

# 3. `bind_tools()` 到底做了什么？

`bind_tools()` 并不会真正执行 Tool。

它的核心作用是：

> **将工具的名称、描述以及参数 Schema 提供给模型。**

例如一个 `read_file` Tool 可以被描述为：

```text
name:
read_file

description:
读取代码仓库中的文件

parameters:
path
start_line
end_line
```

执行：

```python
llm.bind_tools(tools)
```

之后，模型在推理时就知道：

> 我现在拥有一个叫 `read_file` 的工具，并且调用它需要提供 `path`、`start_line`、`end_line` 等参数。

因此：

```python
ai = self.llm_with_tools.invoke(messages)
```

之后，模型可能不会直接回答用户，而是产生一个结构化 Tool Call：

```python
{
    "name": "read_file",
    "args": {
        "path": "src/codebase_agent/agent.py",
        "start_line": 100,
        "end_line": 150
    },
    "id": "call_abc123"
}
```

此时模型只完成了一件事情：

> **提出一个 Action。**

它并没有真的打开 `agent.py`。

---

# 4. `bind_tools()` 和 `invoke()` 返回的东西不同

这里很容易混淆。

下面这行：

```python
self.llm_with_tools = llm.bind_tools(tools)
```

得到的是：

> 一个绑定了 Tool Schema 的模型调用对象。

而：

```python
ai = self.llm_with_tools.invoke(messages)
```

得到的是：

> 当前这一轮模型推理产生的 `AIMessage`。

所以完整关系应该是：

```text
LLM
 │
 │ bind_tools(tools)
 ▼
LLM With Tools
 │
 │ invoke(messages)
 ▼
AIMessage
```

因此：

- `llm_with_tools` 是可以被调用的模型对象；
- `AIMessage` 是某一次具体模型推理的输出。

---

# 5. 模型如何表示“我要调用 Tool”？

`AIMessage` 中可能包含：

```python
ai.tool_calls
```

Agent Runtime 正是通过它判断模型接下来想做什么。

如果：

```python
ai.tool_calls
```

非空，说明模型认为：

> 当前信息还不够，需要调用外部工具继续获取信息。

例如：

```text
AIMessage
├── content: "我需要先查看 agent.py"
│
└── tool_calls:
      └── read_file(
            path="src/codebase_agent/agent.py"
          )
```

这时 Runtime 会继续执行 Tool。

---

# 6. 模型什么时候认为可以回答？

Agent Loop 中一个非常重要的判断是：

```python
if not ai.tool_calls:
    final = ai
    break
```

也就是说：

```text
ai.tool_calls != []
```

代表：

> 模型还希望继续执行动作。

而：

```text
ai.tool_calls == []
```

代表：

> 模型认为当前信息已经足够，不再请求任何外部 Tool，可以结束调查阶段。

例如：

```text
Iteration 1

LLM
↓
retrieve_context
↓
返回相关代码


Iteration 2

LLM
↓
read_file(agent.py)
↓
返回完整源码


Iteration 3

LLM
↓
没有 Tool Call
↓
准备回答
```

所以在正常情况下：

> **Agent Loop 是否结束，是由模型是否继续产生 Tool Call 决定的。**

---

# 7. Iteration 和 Tool Call 不是一回事

项目中还存在两个不同的限制：

```text
max_iterations
max_tool_calls
```

它们不能混为一谈。

## 7.1 Iteration

`iteration` 表示：

> **LLM 做了多少轮决策。**

通常每执行一次：

```python
self.llm_with_tools.invoke(messages)
```

就相当于进行了一轮新的 Agent Decision。

---

## 7.2 Tool Call

Tool Call 表示：

> **Runtime 实际执行了多少次外部工具。**

模型一次推理可以同时产生多个 Tool Call。

例如：

```text
Iteration 1

LLM
├── read_file(agent.py)
├── read_file(rag.py)
└── search_code("Retriever")
```

这时候：

```text
iterations = 1
tool_calls = 3
```

所以：

```text
max_iterations
```

限制的是：

> 模型最多进行多少轮决策。

而：

```text
max_tool_calls
```

限制的是：

> Agent 最多执行多少次外部动作。

两者共同用于防止：

- Agent 无限循环；
- 无意义地反复搜索；
- Tool 调用次数失控；
- Token 和 API 成本不断增加。

---

# 8. 真正执行 Tool 的不是 LLM

假设模型产生：

```python
{
    "name": "read_file",
    "args": {
        "path": "src/codebase_agent/agent.py"
    },
    "id": "call_123"
}
```

之后会进入 Agent Runtime 的 `_execute()`。

首先根据 Tool 名称查找真正的 Python Tool：

```python
tool = self.tools.get(name)
```

然后真正执行 Tool 的关键代码是：

```python
output = str(tool.invoke(args))
```

直到这一行：

> 文件读取、代码搜索、Retriever 查询等实际行为才真正发生。

所以整体链路是：

```text
LLM
↓
产生结构化 Tool Call

Python Runtime
↓
解析 name / args / id

self.tools[name]
↓
找到真正的 Tool

tool.invoke(args)
↓
真正执行 Python 代码
```

因此：

> **LLM 负责生成 Action，Runtime 负责执行 Action。**

---

# 9. 为什么既要有 `llm_with_tools`，又要有 `self.tools`？

这是整个 Tool Calling 架构里一个非常重要的问题。

## `llm_with_tools`

负责：

> 告诉模型当前有哪些工具，以及它们如何调用。

可以理解成：

> **Capability Description**

---

## `self.tools`

负责：

> 在模型产生 Tool Call 后找到真正的 Python Tool，并执行它。

可以理解成：

> **Runtime Implementation**

---

所以：

```text
llm_with_tools
```

解决的是：

> 模型“知道自己可以做什么”。

而：

```text
self.tools
```

解决的是：

> Runtime“知道模型提出动作后到底怎么执行”。

---

# 10. Tool Call 中的 `name`、`args` 和 `id`

一个 Tool Call 通常至少包含：

```python
name
args
id
```

---

## 10.1 `name`

表示调用哪个 Tool。

例如：

```python
name = "read_file"
```

Runtime 就可以执行：

```python
self.tools["read_file"]
```

---

## 10.2 `args`

表示 Tool 需要使用的参数。

例如：

```python
{
    "path": "src/codebase_agent/agent.py",
    "start_line": 100,
    "end_line": 150
}
```

最终：

```python
tool.invoke(args)
```

会根据这些参数执行真正的 Tool。

---

## 10.3 `id`

`id` 的作用不是简单记录 Tool 名称。

它更准确的含义是：

> **一次 Tool Call 的唯一关联标识（Correlation ID）。**

例如模型一次产生三个 Tool Call：

```text
call_A → read_file(agent.py)

call_B → read_file(rag.py)

call_C → search_code("Retriever")
```

Runtime 执行之后分别返回：

```text
ToolMessage(tool_call_id=call_A)

ToolMessage(tool_call_id=call_B)

ToolMessage(tool_call_id=call_C)
```

这样下一轮模型就能够知道：

> 每一个 Tool Result 分别对应自己之前提出的哪一个 Tool Call。

因此 `id` 与传统软件系统中的：

```text
request_id
task_id
transaction_id
```

非常相似。

它不仅用于区分不同轮次。

即使**同一轮模型同时调用多个相同 Tool**，仍然需要不同的 `id`。

---

# 11. 为什么 Tool 执行需要 `try / except`？

Tool 的执行通常会被放在：

```python
try:
    output = str(tool.invoke(args))
except Exception as exc:
    ...
```

原因在于：

> **Tool Failure 并不一定意味着整个 Agent Request 都失败了。**

例如：

```text
LLM
↓
read_file("foo.py")
```

但：

```text
foo.py 不存在
```

如果异常直接向上传播：

```text
FileNotFoundError
↓
整个 Agent 直接退出
```

那么模型一次很小的决策错误就会导致整个任务失败。

更加合理的方式是：

```text
LLM
↓
read_file(foo.py)

Runtime
↓
文件不存在

捕获异常
↓
ERROR Observation

重新交给 LLM
↓
模型调整策略

search_code(...)
```

因此可以得到一个重要的 Agent 工程原则：

> **Tool Failure 是一次 Agent trajectory 中的局部执行失败，不一定属于整个 Request 的致命失败。**

Agent 可以根据错误信息：

- 修改参数；
- 重新调用 Tool；
- 换一个 Tool；
- 使用已有 Evidence；
- 或直接结束调查。

---

# 12. `ToolCallRecord` 和 `ToolMessage` 为什么同时存在？

一次 Tool 执行完成以后，项目中会产生两个很容易混淆的对象：

```text
ToolCallRecord
ToolMessage
```

但它们面向的是两个不同对象。

最简单的记忆方式是：

```text
ToolMessage
=
给模型看的 Observation


ToolCallRecord
=
给 Runtime 看的 Execution Record
```

---

## 12.1 `ToolMessage`

`ToolMessage` 会被放回：

```python
messages
```

用于下一轮 LLM 推理。

它告诉模型：

> 你刚才请求的 Tool 已经执行，这里是执行结果。

因此它属于：

> **LLM Conversation State**

---

## 12.2 `ToolCallRecord`

`ToolCallRecord` 服务于程序自身。

通常用于保存：

```text
Tool Name
Arguments
Success / Failure
Duration
Output
Error
```

然后用于：

- Trace
- Metrics
- Evidence
- Evaluation
- Latency Analysis
- Error Tracking
- Guard

因此它属于：

> **Agent Runtime State**

---

# 13. 为什么不能只保留 `ToolMessage`？

如果只保留：

```text
ToolMessage
```

Agent 的 LLM 对话理论上仍然可以继续。

因为模型仍然能够看到 Tool Result。

但是 Runtime 会失去结构化的执行记录。

例如无法方便地知道：

```text
调用了多少次Tool？

调用成功多少次？

每个Tool耗时多少？

有哪些Evidence来自成功Tool？

是否应该降低Confidence？
```

对于 `codebase-agent` 来说，还会影响：

```text
Trace
Metrics
Evidence
Evaluation
Guard
```

甚至可能导致 Guard 错误判断：

> 没有成功获取 Repository Evidence。

所以：

```text
只有 ToolMessage

LLM Loop              ✅
模型看到 Tool Result   ✅

Runtime Trace          ❌
Metrics                ❌
Evidence Records       ❌
Evaluation             ❌
Guard                  ❌ / 可能误判
```

---

# 14. 为什么不能只保留 `ToolCallRecord`？

如果只保存：

```text
ToolCallRecord
```

Python Runtime 自己当然知道：

```text
read_file执行成功
结果是什么
耗时是多少
```

但是：

> **LLM 并不知道。**

因为这些信息没有进入：

```python
messages
```

下一轮 LLM 看不到 Tool Result。

于是整个 Agent Loop 无法继续。

---

# 15. 为什么 Tool 执行完之后还必须重新调用 LLM？

假设：

```text
User:
Evidence Guard 是怎么实现的？

LLM:
我要读取 agent.py。
```

Runtime 执行：

```python
tool.invoke(...)
```

拿到了 `agent.py` 的内容。

但是这时候还不能直接结束。

因为模型还不知道 Tool 执行的结果。

因此 Runtime 必须构造：

```python
ToolMessage(...)
```

并执行：

```python
messages.append(tool_message)
```

然后再次：

```python
self.llm_with_tools.invoke(messages)
```

此时模型看到的上下文大致是：

```text
User:
Evidence Guard 是怎么实现的？


AI:
我要调用 read_file(agent.py)

tool_call_id = call_001


Tool:
call_001 的执行结果：

......agent.py源码......
```

这样模型才能继续判断：

> 我已经看到 `_apply_guards()` 了。

然后进一步决定：

- 是否需要继续读取其他文件；
- 是否需要搜索其他 Symbol；
- 或者当前 Evidence 已经足够，可以回答。

---

# 16. Agent Loop 的本质：Action → Observation

因此完整的 Tool Calling 可以抽象为：

```text
       LLM
        │
        │ Action
        ▼
    Tool Call
        │
        ▼
 Python Runtime
        │
        │ Observation
        ▼
  ToolMessage
        │
        ▼
       LLM
        │
        ▼
   Next Action
```

即：

> **Action → Observation → Action → Observation**

模型负责：

> 决定下一步做什么。

Runtime 负责：

> 真正执行这个动作。

然后通过 ToolMessage：

> 把环境反馈重新交给模型。

这就是 Agent Loop 最基础的运行机制。

---

# 17. Message History 实际上是一种通信协议

继续抽象之后，会发现模型和 Runtime 本身拥有完全不同的能力。

LLM 可以：

```text
理解自然语言
分析问题
制定计划
选择Tool
根据Evidence继续决策
```

但是它不能直接：

```text
读取本地文件
访问数据库
执行Python函数
访问操作系统
查询向量数据库
```

Python Runtime 可以真正执行这些操作，但是它又不具备 LLM 的语言理解和动态决策能力。

因此两者通过：

```text
AIMessage
Tool Call
ToolMessage
```

完成通信。

可以把整个 Agent 理解为三个角色：

```text
Model
=
决策者


Tool Runtime
=
执行者


Message History
=
决策者和执行者之间的通信协议与状态载体
```

也就是：

```text
Model:
“我要执行A”

        ↓

Runtime:
执行A

        ↓

Runtime:
“A的执行结果是B”

        ↓

Model:
根据B决定下一步执行C
```

---

# 18. Tool Calling 和 Retry 不是一回事

项目中还有一个容易误解的：

```python
for attempt in range(2):
```

一开始很容易把它理解成：

> LLM 请求失败后重试两次。

实际上并不是。

---

## 18.1 正常 Agent Loop

正常的 Agent Loop 是：

```text
LLM
↓
Tool
↓
LLM
↓
Tool
↓
LLM
↓
Final
```

这是正常的：

> **Reasoning / Acting Iteration**

不是 Retry。

---

# 19. Evidence Nudge：行为纠正，而不是 API Retry

假设模型收到问题以后没有调用任何 Tool，而是直接回答：

```text
User:
这个项目的Evidence Guard怎么实现？

LLM:
Evidence Guard通常通过……
```

虽然这个答案可能听起来正确，但问题在于：

> 模型根本没有读取当前 Repository。

对于 Codebase Agent 来说，这是不允许的。

因为它的任务不是：

> 回答一般的软件知识。

而是：

> 根据当前代码仓库中的真实 Evidence 回答问题。

因此外层 Runtime 会检测：

```text
records == []
```

如果：

```text
require_evidence = True
```

并且模型已经给出了 Final Answer，那么 Runtime 会插入：

```text
EVIDENCE_NUDGE
```

提醒模型：

> 你刚才没有查看 Repository，请先获取 Evidence。

然后让 Agent 再运行一次。

所以：

```text
第一次Agent Loop
↓
模型没有Tool Call
↓
直接回答

Runtime检测到：
没有Evidence

↓
EVIDENCE_NUDGE

第二次Agent Loop
↓
要求模型先查Repository
```

这个过程更准确的名称应该是：

> **Corrective Retry**

或者：

> **Evidence Enforcement**

而不是传统意义上的 API Retry。

---

# 20. API Retry 和 Evidence Retry 的区别

真正的 API Retry 通常解决：

```text
HTTP 429

Timeout

Connection Reset

HTTP 5xx
```

例如：

```text
调用LLM API
↓
Timeout
↓
等待
↓
重新请求
```

它解决的是：

> **调用失败。**

而 Evidence Nudge 的情况是：

```text
LLM API调用成功
↓
模型正常返回结果
↓
但是模型行为违反Agent规则
↓
Runtime追加Feedback
↓
重新让模型决策
```

它解决的是：

> **模型行为不符合 Agent 要求。**

因此两者本质不同。

---

# 21. 为什么 Evidence Nudge 只执行一次？

如果模型第二次仍然不调用 Tool：

```text
EVIDENCE_NUDGE
↓
LLM
↓
仍然直接回答
```

系统不会：

```text
你去查
↓
模型不查
↓
你去查
↓
模型不查
↓
无限循环……
```

因为这样会导致：

- Token 消耗持续增加；
- API Cost 增加；
- Latency 增长；
- Agent 可能进入 Infinite Loop。

因此系统只给模型一次行为纠正机会。

如果仍然没有 Evidence：

> 后面的 Guard 会负责降低答案可信度。

---

# 22. Prompt、Correction 和 Guard 三层约束

这个设计体现出了一个很重要的 Agent Engineering 思想：

仅仅依靠 Prompt：

```text
“回答问题之前必须先读取Repository”
```

并不可靠。

因为 Prompt 本质上属于：

> **Soft Constraint**

模型仍然可能不遵守。

因此项目增加第二层：

```text
Runtime检测没有Evidence
↓
Evidence Nudge
```

进行行为纠正。

如果模型仍然不遵守，则进入第三层：

```text
Output Guard
↓
降低Confidence
↓
添加Warning
```

于是形成：

```text
Prompt Constraint
        ↓
Runtime Correction
        ↓
Output Guard
```

也可以概括为：

> **Prompt → Correction → Guard**

相比只在 Prompt 中写：

```text
“禁止幻觉”
```

这种设计更加可靠。

---

# 23. Agent Tool Calling 的完整流程

最终可以把整个机制压缩成：

```text
build_tools()
↓
构造真正的Python Tools


bind_tools()
↓
把Tool Schema提供给LLM


LLM.invoke()
↓
产生AIMessage


ai.tool_calls
↓
模型提出Action


_execute()
↓
Runtime解析
name / args / id


self.tools[name]
↓
找到真正的Python Tool


tool.invoke(args)
↓
执行Tool


ToolCallRecord
↓
保存Runtime执行状态


ToolMessage
↓
把Observation反馈给模型


messages.append()
↓
进入下一轮LLM决策


直到：

ai.tool_calls == []

↓
Agent调查阶段结束
```

---

# 24. 最终理解

Tool Calling 并不是：

> “LLM 自己会调用 Python 函数。”

更加准确的描述应该是：

> **LLM 根据 Tool Schema 生成结构化动作请求，Agent Runtime 解析并执行这个动作，然后通过 ToolMessage 将环境反馈重新加入模型上下文，由模型继续进行下一轮决策。**

因此，一个真正完整的 Agent 并不只是：

```python
llm.bind_tools(tools)
```

真正决定 Agent 行为的是围绕 LLM 建立起来的一整套：

```text
Agent Loop
Tool Runtime
Message State
Execution Record
Iteration Budget
Tool Budget
Error Handling
Evidence
Correction
Guard
Evaluation
```

这些模块共同构成了 Agent 的运行环境。

从这个角度来看：

> **LLM 更像 Agent 的决策核心，而 Agent Harness / Runtime 才是真正让整个 Agent 系统稳定运行的基础设施。**

这也是后续继续学习 Agent Harness Engineering 时需要重点关注的方向。