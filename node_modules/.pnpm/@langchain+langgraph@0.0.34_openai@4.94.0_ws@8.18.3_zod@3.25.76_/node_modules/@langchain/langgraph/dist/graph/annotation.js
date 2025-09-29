import { BinaryOperatorAggregate } from "../channels/binop.js";
import { LastValue } from "../channels/last_value.js";
export class AnnotationRoot {
    constructor(s) {
        Object.defineProperty(this, "lc_graph_name", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: "AnnotationRoot"
        });
        Object.defineProperty(this, "spec", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.spec = s;
    }
}
export function Annotation(annotation) {
    if (annotation) {
        return getChannel(annotation);
    }
    else {
        // @ts-expect-error - Annotation without reducer
        return new LastValue();
    }
}
Annotation.Root = (sd) => new AnnotationRoot(sd);
export function getChannel(reducer) {
    if (typeof reducer === "object" &&
        reducer &&
        "reducer" in reducer &&
        reducer.reducer) {
        return new BinaryOperatorAggregate(reducer.reducer, reducer.default);
    }
    if (typeof reducer === "object" &&
        reducer &&
        "value" in reducer &&
        reducer.value) {
        return new BinaryOperatorAggregate(reducer.value, reducer.default);
    }
    // @ts-expect-error - Annotation without reducer
    return new LastValue();
}
