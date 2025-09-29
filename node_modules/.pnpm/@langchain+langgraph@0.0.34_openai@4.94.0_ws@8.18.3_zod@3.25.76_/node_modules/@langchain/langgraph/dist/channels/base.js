import { deepCopy } from "../checkpoint/base.js";
import { uuid6 } from "../checkpoint/id.js";
import { EmptyChannelError } from "../errors.js";
export class BaseChannel {
    constructor() {
        Object.defineProperty(this, "ValueType", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "UpdateType", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
    }
}
export function emptyChannels(channels, checkpoint) {
    const newChannels = {};
    for (const k in channels) {
        if (Object.prototype.hasOwnProperty.call(channels, k)) {
            const channelValue = checkpoint.channel_values[k];
            newChannels[k] = channels[k].fromCheckpoint(channelValue);
        }
    }
    return newChannels;
}
export function createCheckpoint(checkpoint, channels, step) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const values = {};
    for (const k of Object.keys(channels)) {
        try {
            values[k] = channels[k].checkpoint();
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
        }
        catch (error) {
            if (error.name === EmptyChannelError.unminifiable_name) {
                // no-op
            }
            else {
                throw error; // Rethrow unexpected errors
            }
        }
    }
    return {
        v: 1,
        id: uuid6(step),
        ts: new Date().toISOString(),
        channel_values: values,
        channel_versions: { ...checkpoint.channel_versions },
        versions_seen: deepCopy(checkpoint.versions_seen),
        pending_sends: checkpoint.pending_sends ?? [],
    };
}
