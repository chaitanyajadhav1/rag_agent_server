import { StructuredToolInterface } from "@langchain/core/tools";
import { BaseMessage } from "@langchain/core/messages";
import { RunnableToolLike } from "@langchain/core/runnables";
import { ToolExecutor } from "./tool_executor.js";
import { CompiledStateGraph } from "../graph/state.js";
import { START } from "../graph/index.js";
export type FunctionCallingExecutorState = {
    messages: Array<BaseMessage>;
};
export declare function createFunctionCallingExecutor<Model extends object>({ model, tools, }: {
    model: Model;
    tools: Array<StructuredToolInterface | RunnableToolLike> | ToolExecutor;
}): CompiledStateGraph<FunctionCallingExecutorState, Partial<FunctionCallingExecutorState>, typeof START | "agent" | "action">;
