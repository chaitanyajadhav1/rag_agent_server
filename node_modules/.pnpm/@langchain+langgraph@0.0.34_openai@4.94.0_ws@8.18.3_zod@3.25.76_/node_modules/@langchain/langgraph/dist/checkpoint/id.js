import { v6 } from "uuid";
export function uuid6(clockseq) {
    return v6({ clockseq });
}
