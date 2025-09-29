"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.InvalidUpdateError = exports.EmptyChannelError = exports.GraphValueError = exports.GraphRecursionError = void 0;
class GraphRecursionError extends Error {
    constructor(message) {
        super(message);
        this.name = "GraphRecursionError";
    }
    static get unminifiable_name() {
        return "GraphRecursionError";
    }
}
exports.GraphRecursionError = GraphRecursionError;
class GraphValueError extends Error {
    constructor(message) {
        super(message);
        this.name = "GraphValueError";
    }
    static get unminifiable_name() {
        return "GraphValueError";
    }
}
exports.GraphValueError = GraphValueError;
class EmptyChannelError extends Error {
    constructor(message) {
        super(message);
        this.name = "EmptyChannelError";
    }
    static get unminifiable_name() {
        return "EmptyChannelError";
    }
}
exports.EmptyChannelError = EmptyChannelError;
class InvalidUpdateError extends Error {
    constructor(message) {
        super(message);
        this.name = "InvalidUpdateError";
    }
    static get unminifiable_name() {
        return "InvalidUpdateError";
    }
}
exports.InvalidUpdateError = InvalidUpdateError;
