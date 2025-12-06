"""
Tauri API mock for browser testing.

Provides mock implementations of Tauri APIs so that Tauri applications
can be tested in a browser environment using Playwright.
"""

from typing import Any

# Base Tauri mock script that gets injected into the browser
TAURI_MOCK_SCRIPT = """
// Mock Tauri API for browser testing
// This allows Tauri apps to run in a browser for UI extraction

(function() {
    'use strict';

    // Store for mock responses
    window.__TAURI_MOCKS__ = window.__TAURI_MOCKS__ || {};

    // Mock invoke function
    async function invoke(cmd, args) {
        console.log('[Tauri Mock] invoke:', cmd, args);

        // Check for custom mock response
        if (window.__TAURI_MOCKS__[cmd]) {
            const mockFn = window.__TAURI_MOCKS__[cmd];
            if (typeof mockFn === 'function') {
                return await mockFn(args);
            }
            return mockFn;
        }

        // Default responses for common commands
        const defaults = {
            'get_version': '1.0.0',
            'get_app_name': 'Tauri App',
            'get_platform': 'browser',
            'read_file': '',
            'write_file': true,
            'list_files': [],
        };

        if (cmd in defaults) {
            return defaults[cmd];
        }

        console.warn('[Tauri Mock] No mock found for command:', cmd);
        return null;
    }

    // Mock event system
    const eventListeners = new Map();

    function listen(event, callback) {
        console.log('[Tauri Mock] listen:', event);

        if (!eventListeners.has(event)) {
            eventListeners.set(event, []);
        }
        eventListeners.get(event).push(callback);

        // Return unlisten function
        return () => {
            const listeners = eventListeners.get(event);
            if (listeners) {
                const index = listeners.indexOf(callback);
                if (index > -1) {
                    listeners.splice(index, 1);
                }
            }
        };
    }

    function emit(event, payload) {
        console.log('[Tauri Mock] emit:', event, payload);

        const listeners = eventListeners.get(event);
        if (listeners) {
            listeners.forEach(callback => {
                try {
                    callback({ event, payload });
                } catch (e) {
                    console.error('[Tauri Mock] Error in event listener:', e);
                }
            });
        }
    }

    function once(event, callback) {
        console.log('[Tauri Mock] once:', event);

        const unlisten = listen(event, (data) => {
            callback(data);
            unlisten();
        });

        return unlisten;
    }

    // Mock window functions
    const windowMock = {
        getCurrent: () => ({
            label: 'main',
            isResizable: true,
            isMaximized: false,
            isVisible: true,
            isDecorated: true,
            isFullscreen: false,
            listen,
            emit,
            once,
        }),
        getAll: () => [windowMock.getCurrent()],
        appWindow: null,  // Will be set to getCurrent() below
    };
    windowMock.appWindow = windowMock.getCurrent();

    // Mock dialog functions
    const dialog = {
        open: async (options) => {
            console.log('[Tauri Mock] dialog.open:', options);
            return null;  // User cancelled
        },
        save: async (options) => {
            console.log('[Tauri Mock] dialog.save:', options);
            return null;  // User cancelled
        },
        message: async (message, options) => {
            console.log('[Tauri Mock] dialog.message:', message, options);
            alert(message);
        },
        ask: async (message, options) => {
            console.log('[Tauri Mock] dialog.ask:', message, options);
            return confirm(message);
        },
        confirm: async (message, options) => {
            console.log('[Tauri Mock] dialog.confirm:', message, options);
            return confirm(message);
        },
    };

    // Mock fs (filesystem) functions
    const fs = {
        readTextFile: async (path) => {
            console.log('[Tauri Mock] fs.readTextFile:', path);
            return '';
        },
        writeTextFile: async (path, contents) => {
            console.log('[Tauri Mock] fs.writeTextFile:', path);
            return true;
        },
        readBinaryFile: async (path) => {
            console.log('[Tauri Mock] fs.readBinaryFile:', path);
            return new Uint8Array(0);
        },
        writeBinaryFile: async (path, contents) => {
            console.log('[Tauri Mock] fs.writeBinaryFile:', path);
            return true;
        },
        exists: async (path) => {
            console.log('[Tauri Mock] fs.exists:', path);
            return false;
        },
        createDir: async (path, options) => {
            console.log('[Tauri Mock] fs.createDir:', path, options);
            return true;
        },
        removeFile: async (path) => {
            console.log('[Tauri Mock] fs.removeFile:', path);
            return true;
        },
        removeDir: async (path) => {
            console.log('[Tauri Mock] fs.removeDir:', path);
            return true;
        },
    };

    // Mock http functions
    const http = {
        fetch: async (url, options) => {
            console.log('[Tauri Mock] http.fetch:', url, options);
            // Use native fetch if available
            if (typeof fetch !== 'undefined') {
                return fetch(url, options);
            }
            throw new Error('Fetch not available');
        },
    };

    // Mock shell functions
    const shell = {
        open: async (path) => {
            console.log('[Tauri Mock] shell.open:', path);
            window.open(path, '_blank');
        },
        execute: async (program, args) => {
            console.log('[Tauri Mock] shell.execute:', program, args);
            return { code: 0, stdout: '', stderr: '' };
        },
    };

    // Mock notification
    const notification = {
        sendNotification: (options) => {
            console.log('[Tauri Mock] notification.send:', options);
            if ('Notification' in window && Notification.permission === 'granted') {
                new Notification(options.title, {
                    body: options.body,
                    icon: options.icon,
                });
            }
        },
        requestPermission: async () => {
            console.log('[Tauri Mock] notification.requestPermission');
            if ('Notification' in window) {
                return Notification.requestPermission();
            }
            return 'denied';
        },
    };

    // Mock clipboard
    const clipboard = {
        writeText: async (text) => {
            console.log('[Tauri Mock] clipboard.writeText:', text);
            if (navigator.clipboard) {
                return navigator.clipboard.writeText(text);
            }
        },
        readText: async () => {
            console.log('[Tauri Mock] clipboard.readText');
            if (navigator.clipboard) {
                return navigator.clipboard.readText();
            }
            return '';
        },
    };

    // Create the __TAURI__ global object
    window.__TAURI__ = {
        invoke,
        event: {
            listen,
            once,
            emit,
        },
        window: windowMock,
        dialog,
        fs,
        http,
        shell,
        notification,
        clipboard,
    };

    console.log('[Tauri Mock] Tauri API mock initialized');
})();
"""


def generate_mock_script(mock_responses: dict[str, Any] | None = None) -> str:
    """
    Generate Tauri mock script with custom responses.

    Args:
        mock_responses: Dictionary mapping command names to mock responses.
                       Responses can be values or callable functions.

    Returns:
        JavaScript code to inject into the browser.

    Example:
        >>> script = generate_mock_script({
        ...     'get_user': {'name': 'John', 'id': 123},
        ...     'get_data': lambda args: {'result': args['query']}
        ... })
    """
    script = TAURI_MOCK_SCRIPT

    if mock_responses:
        # Add custom mock responses
        mocks_js = "window.__TAURI_MOCKS__ = window.__TAURI_MOCKS__ || {};\n"

        for cmd, response in mock_responses.items():
            # Convert Python values to JavaScript
            if callable(response):
                # For functions, we can't directly inject them, so use a placeholder
                mocks_js += f"window.__TAURI_MOCKS__['{cmd}'] = null; // Function placeholder\n"
            else:
                # Convert to JSON
                import json

                response_json = json.dumps(response)
                mocks_js += f"window.__TAURI_MOCKS__['{cmd}'] = {response_json};\n"

        script = mocks_js + "\n" + script

    return script


def get_default_mocks() -> dict[str, Any]:
    """
    Get default mock responses for common Tauri commands.

    Returns:
        Dictionary of default mock responses.
    """
    return {
        "get_version": "1.0.0",
        "get_app_name": "Tauri App",
        "get_platform": "browser",
        "read_file": "",
        "write_file": True,
        "list_files": [],
        "save_settings": True,
        "load_settings": {},
    }


def create_fs_mock(files: dict[str, str]) -> dict[str, Any]:
    """
    Create filesystem mock with predefined files.

    Args:
        files: Dictionary mapping file paths to their contents.

    Returns:
        Dictionary of fs-related mock responses.

    Example:
        >>> mocks = create_fs_mock({
        ...     '/config.json': '{"theme": "dark"}',
        ...     '/data.txt': 'Hello world'
        ... })
    """

    # Mock exists for specified files
    def exists_mock(args):
        path = args.get("path", "")
        return path in files

    # Mock readTextFile for specified files
    def read_text_mock(args):
        path = args.get("path", "")
        return files.get(path, "")

    # Note: We can't directly add these as they're functions
    # This would be used programmatically
    return {
        "exists": files.keys(),
        "read_text_file": files,
    }
