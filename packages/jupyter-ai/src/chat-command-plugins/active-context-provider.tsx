/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import type { Contents } from '@jupyterlab/services';
import type { DocumentRegistry } from '@jupyterlab/docregistry';
import {
  IChatCommandProvider,
  IChatCommandRegistry,
  IInputModel
} from '@jupyter/chat';
import { getEditor, getCellIndex } from '../utils';
import { DocumentWidget } from '@jupyterlab/docregistry';
import { NotebookPanel } from '@jupyterlab/notebook';

const ACTIVE_CONTEXT_PROVIDER_ID = '@jupyter-ai/core:active-context-provider';

interface ActiveCellMetadata {
  cellId: string;
  cellIndex: number;
  cellType: 'code' | 'markdown' | 'raw';
  source?: string; // Include source for context (first few lines)
  executionCount?: number | null;
  hasOutput?: boolean;
}

interface ActiveFileMetadata {
  path: string;
  relativePath: string;
  mimeType: string;
  size?: number;
  language?: string;
  cursorPosition?: { line: number; column: number };
  isActive: boolean;
  // Notebook-specific metadata
  isNotebook?: boolean;
  activeCell?: ActiveCellMetadata;
  totalCells?: number;
}

interface ActiveContextMetadata {
  activeFile?: ActiveFileMetadata;
  openTabs: ActiveFileMetadata[];
  workspaceRoot: string;
  timestamp: number;
}

/**
 * A command provider that automatically captures metadata about active files
 * and open tabs, passing this context to Claude without including file content.
 * Claude can then use its existing Read/Grep/LS tools based on this context.
 */
export class ActiveContextProvider implements IChatCommandProvider {
  public id: string = ACTIVE_CONTEXT_PROVIDER_ID;

  constructor(
    private app: JupyterFrontEnd,
    private contentsManager: Contents.IManager,
    private docRegistry: DocumentRegistry
  ) {}

  async listCommandCompletions(): Promise<any[]> {
    // This provider doesn't offer explicit commands, it runs automatically
    return [];
  }

  async onSubmit(inputModel: IInputModel): Promise<void> {
    try {
      const activeContext = await this.captureActiveContext();
      
      if (activeContext) {
        // Add active context metadata as a special attachment
        inputModel.addAttachment?.({
          type: 'file', // Use existing attachment type
          value: `__active_context__:${JSON.stringify(activeContext)}`
        });
      }
    } catch (error) {
      console.warn('Failed to capture active context:', error);
      // Don't throw - we don't want to break chat if context capture fails
    }
  }

  private async captureActiveContext(): Promise<ActiveContextMetadata | null> {
    const currentWidget = this.app.shell.currentWidget;
    const workspaceRoot = await this.getWorkspaceRoot();
    
    const openTabs = await this.getOpenTabsMetadata();
    const activeFile = currentWidget ? await this.getFileMetadata(currentWidget, true) : null;

    return {
      activeFile: activeFile || undefined,
      openTabs,
      workspaceRoot,
      timestamp: Date.now()
    };
  }

  private async getOpenTabsMetadata(): Promise<ActiveFileMetadata[]> {
    const tabs: ActiveFileMetadata[] = [];
    const currentWidget = this.app.shell.currentWidget;

    // Iterate through all document widgets in the main area
    const widgets = Array.from(this.app.shell.widgets('main'));
    for (const widget of widgets) {
      if (widget instanceof DocumentWidget) {
        const metadata = await this.getFileMetadata(widget, widget === currentWidget);
        if (metadata) {
          tabs.push(metadata);
        }
      }
    }

    return tabs;
  }

  private async getFileMetadata(widget: any, isActive: boolean): Promise<ActiveFileMetadata | null> {
    if (!(widget instanceof DocumentWidget)) {
      return null;
    }

    const context = widget.context;
    if (!context || !context.path) {
      return null;
    }

    try {
      // Get file info from contents manager for size and other metadata
      const fileModel = await this.contentsManager.get(context.path, { content: false });
      
      // Get relative path
      const workspaceRoot = await this.getWorkspaceRoot();
      const relativePath = context.path.startsWith(workspaceRoot) 
        ? context.path.slice(workspaceRoot.length + 1)
        : context.path;

      // Get language info
      const fileType = this.docRegistry.getFileTypeForModel(fileModel);
      const language = this.getLanguageFromFileType(fileType, context.path);

      // Check if this is a notebook
      const isNotebook = context.path.endsWith('.ipynb');
      
      // Get cursor position if this is the active file
      let cursorPosition: { line: number; column: number } | undefined;
      if (isActive) {
        cursorPosition = this.getCursorPosition(widget);
      }

      // Get notebook-specific metadata if this is a notebook
      let activeCell: ActiveCellMetadata | undefined;
      let totalCells: number | undefined;
      
      if (isNotebook && isActive) {
        const notebookData = await this.getNotebookMetadata(widget);
        activeCell = notebookData?.activeCell;
        totalCells = notebookData?.totalCells;
      }

      return {
        path: context.path,
        relativePath,
        mimeType: fileModel.mimetype || 'text/plain',
        size: fileModel.size,
        language,
        cursorPosition,
        isActive,
        isNotebook,
        activeCell,
        totalCells
      };
    } catch (error) {
      console.warn(`Failed to get metadata for ${context.path}:`, error);
      return null;
    }
  }

  private getCursorPosition(widget: DocumentWidget): { line: number; column: number } | undefined {
    const editor = getEditor(widget);
    if (!editor) {
      return undefined;
    }

    const cursor = editor.getCursorPosition();
    return {
      line: cursor.line,
      column: cursor.column
    };
  }

  private getLanguageFromFileType(fileType: DocumentRegistry.IFileType, path: string): string | undefined {
    // Try to get language from file type
    if (fileType.name && fileType.name !== 'text') {
      return fileType.name;
    }

    // Fall back to file extension
    const extension = path.split('.').pop()?.toLowerCase();
    const languageMap: { [key: string]: string } = {
      'py': 'python',
      'js': 'javascript',
      'ts': 'typescript',
      'tsx': 'typescript',
      'jsx': 'javascript',
      'java': 'java',
      'cpp': 'cpp',
      'c': 'c',
      'h': 'c',
      'hpp': 'cpp',
      'rs': 'rust',
      'go': 'go',
      'rb': 'ruby',
      'php': 'php',
      'sh': 'bash',
      'sql': 'sql',
      'md': 'markdown',
      'yaml': 'yaml',
      'yml': 'yaml',
      'json': 'json',
      'xml': 'xml',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'less': 'less'
    };

    return extension ? languageMap[extension] : undefined;
  }

  private async getNotebookMetadata(widget: DocumentWidget): Promise<{activeCell?: ActiveCellMetadata; totalCells?: number} | null> {
    try {
      // Check if this is a NotebookPanel
      if (!(widget instanceof NotebookPanel)) {
        return null;
      }

      const notebook = widget.content;
      if (!notebook) {
        return null;
      }

      // Get active cell
      const activeCell = notebook.activeCell;
      if (!activeCell) {
        return { totalCells: notebook.widgets.length };
      }

      // Get cell metadata
      const cellId = activeCell.model.id;
      const cellIndex = getCellIndex(notebook, cellId);
      const cellType = activeCell.model.type as 'code' | 'markdown' | 'raw';
      
      // Get source content (first few lines for context, not the full content)
      const fullSource = activeCell.model.sharedModel.source;
      const sourceLines = fullSource.split('\n');
      const maxLines = 10; // Limit to first 10 lines for context
      const source = sourceLines.length > maxLines 
        ? sourceLines.slice(0, maxLines).join('\n') + '\n...(truncated)'
        : fullSource;

      // Get execution info for code cells
      let executionCount: number | null | undefined;
      let hasOutput = false;
      
      if (cellType === 'code') {
        const codeCell = activeCell.model;
        if ('executionCount' in codeCell) {
          executionCount = (codeCell as any).executionCount;
        }
        if ('outputs' in codeCell) {
          hasOutput = (codeCell as any).outputs?.length > 0;
        }
      }

      const activeCellMetadata: ActiveCellMetadata = {
        cellId,
        cellIndex,
        cellType,
        source,
        executionCount,
        hasOutput
      };

      return {
        activeCell: activeCellMetadata,
        totalCells: notebook.widgets.length
      };

    } catch (error) {
      console.warn('Failed to get notebook metadata:', error);
      return null;
    }
  }

  private async getWorkspaceRoot(): Promise<string> {
    try {
      // Try to get the server root path
      const serverSettings = this.app.serviceManager.serverSettings;
      return serverSettings.baseUrl.replace(/\/api\/.*$/, '') || '';
    } catch (error) {
      console.warn('Failed to get workspace root:', error);
      return '';
    }
  }
}

export const activeContextPlugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyter-ai/core:active-context-plugin',
  description: 'Automatically captures active file context metadata for Claude.',
  autoStart: true,
  requires: [IChatCommandRegistry],
  activate: (app: JupyterFrontEnd, registry: IChatCommandRegistry) => {
    const { serviceManager, docRegistry } = app;
    registry.addProvider(
      new ActiveContextProvider(app, serviceManager.contents, docRegistry)
    );
  }
};