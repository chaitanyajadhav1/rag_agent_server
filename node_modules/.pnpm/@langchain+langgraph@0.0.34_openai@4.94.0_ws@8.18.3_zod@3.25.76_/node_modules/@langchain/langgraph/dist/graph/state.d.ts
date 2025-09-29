import { Runnable, RunnableConfig, RunnableLike } from "@langchain/core/runnables";
import { BaseChannel } from "../channels/base.js";
import { END, CompiledGraph, Graph, START, Branch } from "./graph.js";
import { BaseCheckpointSaver } from "../checkpoint/base.js";
import { All } from "../pregel/types.js";
import { AnnotationRoot, SingleReducer, StateDefinition, StateType, UpdateType } from "./annotation.js";
export type ChannelReducers<Channels extends object> = {
    [K in keyof Channels]: SingleReducer<Channels[K], any>;
};
export interface StateGraphArgs<Channels extends object | unknown> {
    channels: Channels extends object ? Channels extends unknown[] ? ChannelReducers<{
        __root__: Channels;
    }> : ChannelReducers<Channels> : ChannelReducers<{
        __root__: Channels;
    }>;
}
export declare class StateGraph<SD extends StateDefinition | unknown, S = SD extends StateDefinition ? StateType<SD> : SD, U = SD extends StateDefinition ? UpdateType<SD> : Partial<S>, N extends string = typeof START> extends Graph<N, S, U> {
    channels: Record<string, BaseChannel>;
    waitingEdges: Set<[N[], N]>;
    constructor(fields: SD extends StateDefinition ? SD | AnnotationRoot<SD> | StateGraphArgs<S> : StateGraphArgs<S>);
    get allEdges(): Set<[string, string]>;
    addNode<K extends string, NodeInput = S>(key: K, action: RunnableLike<NodeInput, U>): StateGraph<SD, S, U, N | K>;
    addEdge(startKey: typeof START | N | N[], endKey: N | typeof END): this;
    compile({ checkpointer, interruptBefore, interruptAfter, }?: {
        checkpointer?: BaseCheckpointSaver;
        interruptBefore?: N[] | All;
        interruptAfter?: N[] | All;
    }): CompiledStateGraph<S, U, N>;
}
export declare class CompiledStateGraph<S, U, N extends string = typeof START> extends CompiledGraph<N, S, U> {
    builder: StateGraph<unknown, S, U, N>;
    attachNode(key: typeof START, node?: never): void;
    attachNode(key: N, node: Runnable<S, U, RunnableConfig>): void;
    attachEdge(start: N | N[] | "__start__", end: N | "__end__"): void;
    attachBranch(start: N | typeof START, name: string, branch: Branch<S, N>): void;
}
