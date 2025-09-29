import { RunnableLike } from "@langchain/core/runnables";
import { BaseChannel } from "../channels/base.js";
import { BinaryOperator, BinaryOperatorAggregate } from "../channels/binop.js";
import { LastValue } from "../channels/last_value.js";
export type SingleReducer<ValueType, UpdateType = ValueType> = {
    reducer: BinaryOperator<ValueType, UpdateType>;
    default?: () => ValueType;
} | {
    /**
     * @deprecated Use `reducer` instead
     */
    value: BinaryOperator<ValueType, UpdateType>;
    default?: () => ValueType;
} | null;
export interface StateDefinition {
    [key: string]: BaseChannel | (() => BaseChannel);
}
type ExtractValueType<C> = C extends BaseChannel ? C["ValueType"] : C extends () => BaseChannel ? ReturnType<C>["ValueType"] : never;
type ExtractUpdateType<C> = C extends BaseChannel ? C["UpdateType"] : C extends () => BaseChannel ? ReturnType<C>["UpdateType"] : never;
export type StateType<SD extends StateDefinition> = {
    [key in keyof SD]: ExtractValueType<SD[key]>;
};
export type UpdateType<SD extends StateDefinition> = {
    [key in keyof SD]?: ExtractUpdateType<SD[key]>;
};
export type NodeType<SD extends StateDefinition> = RunnableLike<StateType<SD>, UpdateType<SD>>;
export declare class AnnotationRoot<SD extends StateDefinition> {
    lc_graph_name: string;
    State: StateType<SD>;
    Update: UpdateType<SD>;
    Node: NodeType<SD>;
    spec: SD;
    constructor(s: SD);
}
export declare function Annotation<ValueType>(): LastValue<ValueType>;
export declare function Annotation<ValueType, UpdateType = ValueType>(annotation: SingleReducer<ValueType, UpdateType>): BinaryOperatorAggregate<ValueType, UpdateType>;
export declare namespace Annotation {
    var Root: <S extends StateDefinition>(sd: S) => AnnotationRoot<S>;
}
export declare function getChannel<V, U = V>(reducer: SingleReducer<V, U>): BaseChannel<V, U>;
export {};
