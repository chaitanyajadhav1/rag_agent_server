"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CompiledGraph = exports.Graph = exports.Branch = exports.END = exports.START = void 0;
/* eslint-disable @typescript-eslint/no-use-before-define */
const runnables_1 = require("@langchain/core/runnables");
const graph_1 = require("@langchain/core/runnables/graph");
const zod_1 = require("zod");
const read_js_1 = require("../pregel/read.cjs");
const index_js_1 = require("../pregel/index.cjs");
const ephemeral_value_js_1 = require("../channels/ephemeral_value.cjs");
const write_js_1 = require("../pregel/write.cjs");
const constants_js_1 = require("../constants.cjs");
const utils_js_1 = require("../utils.cjs");
const errors_js_1 = require("../errors.cjs");
exports.START = "__start__";
exports.END = "__end__";
class Branch {
    constructor(options) {
        Object.defineProperty(this, "condition", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "ends", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.condition = options.path;
        this.ends = Array.isArray(options.pathMap)
            ? options.pathMap.reduce((acc, n) => {
                acc[n] = n;
                return acc;
            }, {})
            : options.pathMap;
    }
    compile(writer, reader) {
        return write_js_1.ChannelWrite.registerWriter(new utils_js_1.RunnableCallable({
            func: (input, config) => this._route(input, config, writer, reader),
        }));
    }
    async _route(input, config, writer, reader) {
        let result = await this.condition(reader ? reader(config) : input, config);
        if (!Array.isArray(result)) {
            result = [result];
        }
        let destinations;
        if (this.ends) {
            // destinations = [r if isinstance(r, Send) else self.ends[r] for r in result]
            destinations = result.map((r) => ((0, constants_js_1._isSend)(r) ? r : this.ends[r]));
        }
        else {
            destinations = result;
        }
        if (destinations.some((dest) => !dest)) {
            throw new Error("Branch condition returned unknown or null destination");
        }
        if (destinations.filter(constants_js_1._isSend).some((packet) => packet.node === exports.END)) {
            throw new errors_js_1.InvalidUpdateError("Cannot send a packet to the END node");
        }
        return writer(destinations);
    }
}
exports.Branch = Branch;
class Graph {
    constructor() {
        Object.defineProperty(this, "nodes", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "edges", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "branches", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "entryPoint", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "compiled", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: false
        });
        Object.defineProperty(this, "supportMultipleEdges", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: false
        });
        this.nodes = {};
        this.edges = new Set();
        this.branches = {};
    }
    warnIfCompiled(message) {
        if (this.compiled) {
            console.warn(message);
        }
    }
    get allEdges() {
        return this.edges;
    }
    addNode(key, action) {
        this.warnIfCompiled(`Adding a node to a graph that has already been compiled. This will not be reflected in the compiled graph.`);
        if (key in this.nodes) {
            throw new Error(`Node \`${key}\` already present.`);
        }
        if (key === exports.END) {
            throw new Error(`Node \`${key}\` is reserved.`);
        }
        this.nodes[key] = (0, runnables_1._coerceToRunnable)(
        // Account for arbitrary state due to Send API
        action);
        return this;
    }
    addEdge(startKey, endKey) {
        this.warnIfCompiled(`Adding an edge to a graph that has already been compiled. This will not be reflected in the compiled graph.`);
        if (startKey === exports.END) {
            throw new Error("END cannot be a start node");
        }
        if (endKey === exports.START) {
            throw new Error("START cannot be an end node");
        }
        if (!this.supportMultipleEdges &&
            Array.from(this.edges).some(([start]) => start === startKey)) {
            throw new Error(`Already found path for ${startKey}`);
        }
        this.edges.add([startKey, endKey]);
        return this;
    }
    addConditionalEdges(source, path, pathMap) {
        const options = typeof source === "object" ? source : { source, path: path, pathMap };
        this.warnIfCompiled("Adding an edge to a graph that has already been compiled. This will not be reflected in the compiled graph.");
        // find a name for condition
        const name = options.path.name || "condition";
        // validate condition
        if (this.branches[options.source] && this.branches[options.source][name]) {
            throw new Error(`Condition \`${name}\` already present for node \`${source}\``);
        }
        // save it
        if (!this.branches[options.source]) {
            this.branches[options.source] = {};
        }
        this.branches[options.source][name] = new Branch(options);
        return this;
    }
    /**
     * @deprecated use `addEdge(START, key)` instead
     */
    setEntryPoint(key) {
        this.warnIfCompiled("Setting the entry point of a graph that has already been compiled. This will not be reflected in the compiled graph.");
        return this.addEdge(exports.START, key);
    }
    /**
     * @deprecated use `addEdge(key, END)` instead
     */
    setFinishPoint(key) {
        this.warnIfCompiled("Setting a finish point of a graph that has already been compiled. This will not be reflected in the compiled graph.");
        return this.addEdge(key, exports.END);
    }
    compile({ checkpointer, interruptBefore, interruptAfter, } = {}) {
        // validate the graph
        this.validate([
            ...(Array.isArray(interruptBefore) ? interruptBefore : []),
            ...(Array.isArray(interruptAfter) ? interruptAfter : []),
        ]);
        // create empty compiled graph
        const compiled = new CompiledGraph({
            builder: this,
            checkpointer,
            interruptAfter,
            interruptBefore,
            autoValidate: false,
            nodes: {},
            channels: {
                [exports.START]: new ephemeral_value_js_1.EphemeralValue(),
                [exports.END]: new ephemeral_value_js_1.EphemeralValue(),
            },
            inputs: exports.START,
            outputs: exports.END,
            streamChannels: [],
            streamMode: "values",
        });
        // attach nodes, edges and branches
        for (const [key, node] of Object.entries(this.nodes)) {
            compiled.attachNode(key, node);
        }
        for (const [start, end] of this.edges) {
            compiled.attachEdge(start, end);
        }
        for (const [start, branches] of Object.entries(this.branches)) {
            for (const [name, branch] of Object.entries(branches)) {
                compiled.attachBranch(start, name, branch);
            }
        }
        return compiled.validate();
    }
    validate(interrupt) {
        // assemble sources
        const allSources = new Set([...this.allEdges].map(([src, _]) => src));
        for (const [start] of Object.entries(this.branches)) {
            allSources.add(start);
        }
        // validate sources
        for (const node of Object.keys(this.nodes)) {
            if (!allSources.has(node)) {
                throw new Error(`Node \`${node}\` is a dead-end`);
            }
        }
        for (const source of allSources) {
            if (source !== exports.START && !(source in this.nodes)) {
                throw new Error(`Found edge starting at unknown node \`${source}\``);
            }
        }
        // assemble targets
        const allTargets = new Set([...this.allEdges].map(([_, target]) => target));
        for (const [start, branches] of Object.entries(this.branches)) {
            for (const branch of Object.values(branches)) {
                if (branch.ends) {
                    for (const end of Object.values(branch.ends)) {
                        allTargets.add(end);
                    }
                }
                else {
                    allTargets.add(exports.END);
                    for (const node of Object.keys(this.nodes)) {
                        if (node !== start) {
                            allTargets.add(node);
                        }
                    }
                }
            }
        }
        // validate targets
        for (const node of Object.keys(this.nodes)) {
            if (!allTargets.has(node)) {
                throw new Error(`Node \`${node}\` is not reachable`);
            }
        }
        for (const target of allTargets) {
            if (target !== exports.END && !(target in this.nodes)) {
                throw new Error(`Found edge ending at unknown node \`${target}\``);
            }
        }
        // validate interrupts
        if (interrupt) {
            for (const node of interrupt) {
                if (!(node in this.nodes)) {
                    throw new Error(`Interrupt node \`${node}\` is not present`);
                }
            }
        }
        this.compiled = true;
    }
}
exports.Graph = Graph;
class CompiledGraph extends index_js_1.Pregel {
    constructor({ builder, ...rest }) {
        super(rest);
        Object.defineProperty(this, "builder", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.builder = builder;
    }
    attachNode(key, node) {
        this.channels[key] = new ephemeral_value_js_1.EphemeralValue();
        this.nodes[key] = new read_js_1.PregelNode({
            channels: [],
            triggers: [],
        })
            .pipe(node)
            .pipe(new write_js_1.ChannelWrite([{ channel: key, value: write_js_1.PASSTHROUGH }], [constants_js_1.TAG_HIDDEN]));
        this.streamChannels.push(key);
    }
    attachEdge(start, end) {
        if (end === exports.END) {
            if (start === exports.START) {
                throw new Error("Cannot have an edge from START to END");
            }
            this.nodes[start].writers.push(new write_js_1.ChannelWrite([{ channel: exports.END, value: write_js_1.PASSTHROUGH }], [constants_js_1.TAG_HIDDEN]));
        }
        else {
            this.nodes[end].triggers.push(start);
            this.nodes[end].channels.push(start);
        }
    }
    attachBranch(start, name, branch) {
        // add hidden start node
        if (start === exports.START && this.nodes[exports.START]) {
            this.nodes[exports.START] = index_js_1.Channel.subscribeTo(exports.START, { tags: [constants_js_1.TAG_HIDDEN] });
        }
        // attach branch writer
        this.nodes[start].pipe(branch.compile((dests) => {
            const writes = dests.map((dest) => {
                if ((0, constants_js_1._isSend)(dest)) {
                    return dest;
                }
                return {
                    channel: dest === exports.END ? exports.END : `branch:${start}:${name}:${dest}`,
                    value: write_js_1.PASSTHROUGH,
                };
            });
            return new write_js_1.ChannelWrite(writes, [constants_js_1.TAG_HIDDEN]);
        }));
        // attach branch readers
        const ends = branch.ends
            ? Object.values(branch.ends)
            : Object.keys(this.nodes);
        for (const end of ends) {
            if (end !== exports.END) {
                const channelName = `branch:${start}:${name}:${end}`;
                this.channels[channelName] =
                    new ephemeral_value_js_1.EphemeralValue();
                this.nodes[end].triggers.push(channelName);
                this.nodes[end].channels.push(channelName);
            }
        }
    }
    /**
     * Returns a drawable representation of the computation graph.
     */
    getGraph(config) {
        const xray = config?.xray;
        const graph = new graph_1.Graph();
        const startNodes = {
            [exports.START]: graph.addNode({
                schema: zod_1.z.any(),
            }, exports.START),
        };
        const endNodes = {
            [exports.END]: graph.addNode({
                schema: zod_1.z.any(),
            }, exports.END),
        };
        for (const [key, node] of Object.entries(this.builder.nodes)) {
            if (config?.xray) {
                const subgraph = isCompiledGraph(node)
                    ? node.getGraph({
                        ...config,
                        xray: typeof xray === "number" && xray > 0 ? xray - 1 : xray,
                    })
                    : node.getGraph(config);
                subgraph.trimFirstNode();
                subgraph.trimLastNode();
                if (Object.keys(subgraph.nodes).length > 1) {
                    const [newEndNode, newStartNode] = graph.extend(subgraph, key);
                    if (newEndNode !== undefined) {
                        endNodes[key] = newEndNode;
                    }
                    if (newStartNode !== undefined) {
                        startNodes[key] = newStartNode;
                    }
                }
                else {
                    const newNode = graph.addNode(node, key);
                    startNodes[key] = newNode;
                    endNodes[key] = newNode;
                }
            }
            else {
                const newNode = graph.addNode(node, key);
                startNodes[key] = newNode;
                endNodes[key] = newNode;
            }
        }
        for (const [start, end] of this.builder.allEdges) {
            graph.addEdge(startNodes[start], endNodes[end]);
        }
        for (const [start, branches] of Object.entries(this.builder.branches)) {
            const defaultEnds = {
                ...Object.fromEntries(Object.keys(this.builder.nodes)
                    .filter((k) => k !== start)
                    .map((k) => [k, k])),
                [exports.END]: exports.END,
            };
            for (const branch of Object.values(branches)) {
                let ends;
                if (branch.ends !== undefined) {
                    ends = branch.ends;
                }
                else {
                    ends = defaultEnds;
                }
                for (const [label, end] of Object.entries(ends)) {
                    graph.addEdge(startNodes[start], endNodes[end], label !== end ? label : undefined, true);
                }
            }
        }
        return graph;
    }
}
exports.CompiledGraph = CompiledGraph;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isCompiledGraph(x) {
    return (
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    typeof x.attachNode === "function" &&
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        typeof x.attachEdge === "function");
}
