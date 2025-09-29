import { BaseMessage } from "@langchain/core/messages";
import { RunnableToolLike } from "@langchain/core/runnables";
import { StructuredToolInterface } from "@langchain/core/tools";
import { RunnableCallable } from "../utils.js";
import { END } from "../graph/graph.js";
import { MessagesState } from "../graph/message.js";
export type ToolNodeOptions = {
    name?: string;
    tags?: string[];
    handleToolErrors?: boolean;
};
export declare class ToolNode<T extends BaseMessage[] | MessagesState> extends RunnableCallable<T, T> {
    /**
    A node that runs the tools requested in the last AIMessage. It can be used
    either in StateGraph with a "messages" key or in MessageGraph. If multiple
    tool calls are requested, they will be run in parallel. The output will be
    a list of ToolMessages, one for each tool call.
    */
    tools: (StructuredToolInterface | RunnableToolLike)[];
    handleToolErrors: boolean;
    constructor(tools: (StructuredToolInterface | RunnableToolLike)[], options?: ToolNodeOptions);
    private run;
}
export declare function toolsCondition(state: BaseMessage[] | MessagesState): "tools" | typeof END;
