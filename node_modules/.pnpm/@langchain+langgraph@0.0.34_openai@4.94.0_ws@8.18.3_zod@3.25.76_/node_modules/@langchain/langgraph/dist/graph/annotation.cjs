"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getChannel = exports.Annotation = exports.AnnotationRoot = void 0;
const binop_js_1 = require("../channels/binop.cjs");
const last_value_js_1 = require("../channels/last_value.cjs");
class AnnotationRoot {
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
exports.AnnotationRoot = AnnotationRoot;
function Annotation(annotation) {
    if (annotation) {
        return getChannel(annotation);
    }
    else {
        // @ts-expect-error - Annotation without reducer
        return new last_value_js_1.LastValue();
    }
}
exports.Annotation = Annotation;
Annotation.Root = (sd) => new AnnotationRoot(sd);
function getChannel(reducer) {
    if (typeof reducer === "object" &&
        reducer &&
        "reducer" in reducer &&
        reducer.reducer) {
        return new binop_js_1.BinaryOperatorAggregate(reducer.reducer, reducer.default);
    }
    if (typeof reducer === "object" &&
        reducer &&
        "value" in reducer &&
        reducer.value) {
        return new binop_js_1.BinaryOperatorAggregate(reducer.value, reducer.default);
    }
    // @ts-expect-error - Annotation without reducer
    return new last_value_js_1.LastValue();
}
exports.getChannel = getChannel;
