import { fileCommandPlugin } from './file-command';
import { slashCommandPlugin } from './slash-commands';
import { activeContextPlugin } from './active-context-provider';

export const chatCommandPlugins = [fileCommandPlugin, slashCommandPlugin, activeContextPlugin];
