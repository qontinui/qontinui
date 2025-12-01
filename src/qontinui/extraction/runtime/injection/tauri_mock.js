/**
 * Tauri API Mock for Browser Testing
 *
 * This script provides mock implementations of Tauri APIs so that Tauri applications
 * can be tested in a browser environment using Playwright.
 *
 * Usage:
 *   await page.addInitScript({ path: 'tauri_mock.js' });
 *
 * Or inject via string:
 *   await page.addInitScript(fs.readFileSync('tauri_mock.js', 'utf8'));
 */

(function() {
    'use strict';

    // Prevent double injection
    if (window.__TAURI__) {
        console.warn('[Tauri Mock] Already initialized, skipping');
        return;
    }

    console.log('[Tauri Mock] Initializing Tauri API mock...');

    // Store for mock responses - can be customized externally
    window.__TAURI_MOCKS__ = window.__TAURI_MOCKS__ || {};

    // ============================================================================
    // Core API - invoke
    // ============================================================================

    /**
     * Mock implementation of Tauri's invoke function
     */
    async function invoke(cmd, args = {}) {
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
            'get_os': 'mock',
            'read_file': '',
            'write_file': true,
            'list_files': [],
            'save_settings': true,
            'load_settings': {},
        };

        if (cmd in defaults) {
            return defaults[cmd];
        }

        console.warn('[Tauri Mock] No mock found for command:', cmd);
        return null;
    }

    // ============================================================================
    // Event System
    // ============================================================================

    const eventListeners = new Map();
    let eventIdCounter = 0;

    /**
     * Listen for events
     */
    function listen(event, callback) {
        console.log('[Tauri Mock] listen:', event);

        const eventId = eventIdCounter++;

        if (!eventListeners.has(event)) {
            eventListeners.set(event, new Map());
        }
        eventListeners.get(event).set(eventId, callback);

        // Return unlisten function
        return async () => {
            const listeners = eventListeners.get(event);
            if (listeners) {
                listeners.delete(eventId);
            }
        };
    }

    /**
     * Emit an event
     */
    function emit(event, payload) {
        console.log('[Tauri Mock] emit:', event, payload);

        const listeners = eventListeners.get(event);
        if (listeners) {
            listeners.forEach(callback => {
                try {
                    callback({
                        event,
                        payload,
                        id: Date.now(),
                    });
                } catch (e) {
                    console.error('[Tauri Mock] Error in event listener:', e);
                }
            });
        }
    }

    /**
     * Listen for event once
     */
    function once(event, callback) {
        console.log('[Tauri Mock] once:', event);

        const unlisten = listen(event, (data) => {
            callback(data);
            unlisten();
        });

        return unlisten;
    }

    // ============================================================================
    // Window API
    // ============================================================================

    const windowMock = {
        /**
         * Get current window
         */
        getCurrent: () => ({
            label: 'main',
            isResizable: () => Promise.resolve(true),
            isMaximized: () => Promise.resolve(false),
            isVisible: () => Promise.resolve(true),
            isDecorated: () => Promise.resolve(true),
            isFullscreen: () => Promise.resolve(false),
            isFocused: () => Promise.resolve(true),
            isMinimized: () => Promise.resolve(false),
            setTitle: (title) => {
                console.log('[Tauri Mock] window.setTitle:', title);
                document.title = title;
                return Promise.resolve();
            },
            show: () => {
                console.log('[Tauri Mock] window.show');
                return Promise.resolve();
            },
            hide: () => {
                console.log('[Tauri Mock] window.hide');
                return Promise.resolve();
            },
            close: () => {
                console.log('[Tauri Mock] window.close');
                return Promise.resolve();
            },
            minimize: () => {
                console.log('[Tauri Mock] window.minimize');
                return Promise.resolve();
            },
            maximize: () => {
                console.log('[Tauri Mock] window.maximize');
                return Promise.resolve();
            },
            unmaximize: () => {
                console.log('[Tauri Mock] window.unmaximize');
                return Promise.resolve();
            },
            toggleMaximize: () => {
                console.log('[Tauri Mock] window.toggleMaximize');
                return Promise.resolve();
            },
            listen,
            emit,
            once,
        }),

        /**
         * Get all windows
         */
        getAll: () => [windowMock.getCurrent()],

        /**
         * Current app window
         */
        appWindow: null,  // Will be set below
    };

    // Set appWindow to current window
    windowMock.appWindow = windowMock.getCurrent();

    // ============================================================================
    // Dialog API
    // ============================================================================

    const dialog = {
        /**
         * Open file dialog
         */
        open: async (options = {}) => {
            console.log('[Tauri Mock] dialog.open:', options);
            // Return null to simulate user cancellation
            return null;
        },

        /**
         * Save file dialog
         */
        save: async (options = {}) => {
            console.log('[Tauri Mock] dialog.save:', options);
            // Return null to simulate user cancellation
            return null;
        },

        /**
         * Show message dialog
         */
        message: async (message, options = {}) => {
            console.log('[Tauri Mock] dialog.message:', message, options);
            alert(message);
        },

        /**
         * Show ask dialog (yes/no)
         */
        ask: async (message, options = {}) => {
            console.log('[Tauri Mock] dialog.ask:', message, options);
            return confirm(message);
        },

        /**
         * Show confirm dialog
         */
        confirm: async (message, options = {}) => {
            console.log('[Tauri Mock] dialog.confirm:', message, options);
            return confirm(message);
        },
    };

    // ============================================================================
    // Filesystem API
    // ============================================================================

    const fs = {
        readTextFile: async (path, options = {}) => {
            console.log('[Tauri Mock] fs.readTextFile:', path, options);
            return '';
        },

        writeTextFile: async (path, contents, options = {}) => {
            console.log('[Tauri Mock] fs.writeTextFile:', path, options);
            return Promise.resolve();
        },

        readBinaryFile: async (path, options = {}) => {
            console.log('[Tauri Mock] fs.readBinaryFile:', path, options);
            return new Uint8Array(0);
        },

        writeBinaryFile: async (path, contents, options = {}) => {
            console.log('[Tauri Mock] fs.writeBinaryFile:', path, options);
            return Promise.resolve();
        },

        exists: async (path, options = {}) => {
            console.log('[Tauri Mock] fs.exists:', path, options);
            return false;
        },

        createDir: async (path, options = {}) => {
            console.log('[Tauri Mock] fs.createDir:', path, options);
            return Promise.resolve();
        },

        removeFile: async (path, options = {}) => {
            console.log('[Tauri Mock] fs.removeFile:', path, options);
            return Promise.resolve();
        },

        removeDir: async (path, options = {}) => {
            console.log('[Tauri Mock] fs.removeDir:', path, options);
            return Promise.resolve();
        },

        readDir: async (path, options = {}) => {
            console.log('[Tauri Mock] fs.readDir:', path, options);
            return [];
        },

        renameFile: async (oldPath, newPath, options = {}) => {
            console.log('[Tauri Mock] fs.renameFile:', oldPath, newPath, options);
            return Promise.resolve();
        },

        copyFile: async (source, destination, options = {}) => {
            console.log('[Tauri Mock] fs.copyFile:', source, destination, options);
            return Promise.resolve();
        },
    };

    // ============================================================================
    // HTTP API
    // ============================================================================

    const http = {
        fetch: async (url, options = {}) => {
            console.log('[Tauri Mock] http.fetch:', url, options);

            // Use native fetch if available
            if (typeof fetch !== 'undefined') {
                const response = await fetch(url, options);
                return {
                    url: response.url,
                    status: response.status,
                    statusText: response.statusText,
                    headers: Object.fromEntries(response.headers.entries()),
                    data: await response.text(),
                };
            }

            throw new Error('Fetch not available in mock environment');
        },
    };

    // ============================================================================
    // Shell API
    // ============================================================================

    const shell = {
        /**
         * Open URL in default browser
         */
        open: async (path) => {
            console.log('[Tauri Mock] shell.open:', path);
            window.open(path, '_blank');
        },

        /**
         * Execute command
         */
        execute: async (program, args = [], options = {}) => {
            console.log('[Tauri Mock] shell.execute:', program, args, options);
            return {
                code: 0,
                stdout: '',
                stderr: '',
            };
        },
    };

    // ============================================================================
    // Notification API
    // ============================================================================

    const notification = {
        /**
         * Send notification
         */
        sendNotification: (options) => {
            console.log('[Tauri Mock] notification.send:', options);

            if ('Notification' in window && Notification.permission === 'granted') {
                new Notification(options.title, {
                    body: options.body,
                    icon: options.icon,
                });
            } else {
                console.warn('[Tauri Mock] Notifications not available or not permitted');
            }
        },

        /**
         * Request notification permission
         */
        requestPermission: async () => {
            console.log('[Tauri Mock] notification.requestPermission');

            if ('Notification' in window) {
                return Notification.requestPermission();
            }

            return 'denied';
        },

        /**
         * Check if notification permission is granted
         */
        isPermissionGranted: async () => {
            console.log('[Tauri Mock] notification.isPermissionGranted');

            if ('Notification' in window) {
                return Notification.permission === 'granted';
            }

            return false;
        },
    };

    // ============================================================================
    // Clipboard API
    // ============================================================================

    const clipboard = {
        writeText: async (text) => {
            console.log('[Tauri Mock] clipboard.writeText:', text);

            if (navigator.clipboard) {
                return navigator.clipboard.writeText(text);
            }

            throw new Error('Clipboard API not available');
        },

        readText: async () => {
            console.log('[Tauri Mock] clipboard.readText');

            if (navigator.clipboard) {
                return navigator.clipboard.readText();
            }

            return '';
        },
    };

    // ============================================================================
    // Path API
    // ============================================================================

    const path = {
        appDir: async () => '/app',
        audioDir: async () => '/audio',
        cacheDir: async () => '/cache',
        configDir: async () => '/config',
        dataDir: async () => '/data',
        desktopDir: async () => '/desktop',
        documentDir: async () => '/documents',
        downloadDir: async () => '/downloads',
        executableDir: async () => '/bin',
        fontDir: async () => '/fonts',
        homeDir: async () => '/home',
        localDataDir: async () => '/local',
        pictureDir: async () => '/pictures',
        publicDir: async () => '/public',
        runtimeDir: async () => '/runtime',
        tempDir: async () => '/tmp',
        templateDir: async () => '/templates',
        videoDir: async () => '/videos',
        resourceDir: async () => '/resources',
        logDir: async () => '/logs',

        join: async (...paths) => paths.join('/'),
        dirname: async (path) => path.split('/').slice(0, -1).join('/'),
        basename: async (path, ext) => {
            const base = path.split('/').pop() || '';
            if (ext && base.endsWith(ext)) {
                return base.slice(0, -ext.length);
            }
            return base;
        },
        extname: async (path) => {
            const base = path.split('/').pop() || '';
            const lastDot = base.lastIndexOf('.');
            return lastDot > 0 ? base.slice(lastDot) : '';
        },
    };

    // ============================================================================
    // Create the __TAURI__ global object
    // ============================================================================

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
        path,

        // Metadata
        __mock__: true,
        __version__: '1.0.0',
    };

    // Make it read-only to prevent accidental overwrites
    Object.freeze(window.__TAURI__);

    console.log('[Tauri Mock] Tauri API mock initialized successfully');
    console.log('[Tauri Mock] Available APIs:', Object.keys(window.__TAURI__));
})();
