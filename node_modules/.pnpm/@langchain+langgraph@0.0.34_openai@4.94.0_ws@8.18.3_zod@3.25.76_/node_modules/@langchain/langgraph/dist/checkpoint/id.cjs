"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.uuid6 = void 0;
const uuid_1 = require("uuid");
function uuid6(clockseq) {
    return (0, uuid_1.v6)({ clockseq });
}
exports.uuid6 = uuid6;
