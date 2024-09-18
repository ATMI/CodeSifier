/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @emails react-core
 * @jest-environment node
 */

'use strict';

let Stream;
let React;
let ReactDOMFizzServer;
let Suspense;

describe('ReactDOMFizzServer', () => {
  beforeEach(() => {
    jest.resetModules();
    React = require('react');
    if (__EXPERIMENTAL__) {
      ReactDOMFizzServer = require('react-dom/server');
    }
    Stream = require('stream');
    Suspense = React.Suspense;
  });

  function getTestWritable() {
    const writable = new Stream.PassThrough();
    writable.setEncoding('utf8');
    const output = {result: '', error: undefined};
    writable.on('data', chunk => {
      output.result += chunk;
    });
    writable.on('error', error => {
      output.error = error;
    });
    const completed = new Promise(resolve => {
      writable.on('finish', () => {
        resolve();
      });
      writable.on('error', () => {
        resolve();
      });
    });
    return {writable, completed, output};
  }

  const theError = new Error('This is an error');
  function Throw() {
    throw theError;
  }
  const theInfinitePromise = new Promise(() => {});
  function InfiniteSuspend() {
    throw theInfinitePromise;
  }

  // @gate experimental
  it('should call renderToPipeableStream', () => {
    const {writable, output} = getTestWritable();
    const {pipe} = ReactDOMFizzServer.renderToPipeableStream(
      <div>hello world</div>,
    );
    pipe(writable);
    jest.runAllTimers();
    expect(output.result).toMatchInlineSnapshot(`"<div>hello world</div>"`);
  });

  // @gate experimental
  it('should emit DOCTYPE at the root of the document', () => {
    const {writable, output} = getTestWritable();
    const {pipe} = ReactDOMFizzServer.renderToPipeableStream(
      <html>
        <body>hello world</body>
      </html>,
    );
    pipe(writable);
    jest.runAllTimers();
    expect(output.result).toMatchInlineSnapshot(
      `"<!DOCTYPE html><html><body>hello world</body></html>"`,
    );
  });

  // @gate experimental
  it('should emit bootstrap script src at the end', () => {
    const {writable, output} = getTestWritable();
    const {pipe} = ReactDOMFizzServer.renderToPipeableStream(
      <div>hello world</div>,
      {
        bootstrapScriptContent: 'INIT();',
        bootstrapScripts: ['init.js'],
        bootstrapModules: ['init.mjs'],
      },
    );
    pipe(writable);
    jest.runAllTimers();
    expect(output.result).toMatchInlineSnapshot(
      `"<div>hello world</div><script>INIT();</script><script src=\\"init.js\\" async=\\"\\"></script><script type=\\"module\\" src=\\"init.mjs\\" async=\\"\\"></script>"`,
    );
  });

  // @gate experimental
  it('should start writing after pipe', () => {
    const {writable, output} = getTestWritable();
    const {pipe} = ReactDOMFizzServer.renderToPipeableStream(
      <div>hello world</div>,
    );
    jest.runAllTimers();
    // First we write our header.
    output.result +=
      '<!doctype html><html><head><title>test</title><head><body>';
    // Then React starts writing.
    pipe(writable);
    expect(output.result).toMatchInlineSnapshot(
      `"<!doctype html><html><head><title>test</title><head><body><div>hello world</div>"`,
    );
  });

  // @gate experimental
  it('emits all HTML as one unit if we wait until the end to start', async () => {
    let hasLoaded = false;
    let resolve;
    const promise = new Promise(r => (resolve = r));
    function Wait() {
      if (!hasLoaded) {
        throw promise;
      }
      return 'Done';
    }
    let isCompleteCalls = 0;
    const {writable, output} = getTestWritable();
    const {pipe} = ReactDOMFizzServer.renderToPipeableStream(
      <div>
        <Suspense fallback="Loading">
          <Wait />
        </Suspense>
      </div>,

      {
        onAllReady() {
          isCompleteCalls++;
        },
      },
    );
    await jest.runAllTimers();
    expect(output.result).toBe('');
    expect(isCompleteCalls).toBe(0);
    // Resolve the loading.
    hasLoaded = true;
    await resolve();

    await jest.runAllTimers();

    expect(output.result).toBe('');
    expect(isCompleteCalls).toBe(1);

    // First we write our header.
    output.result +=
      '<!doctype html><html><head><title>test</title><head><body>';
    // Then React starts writing.
    pipe(writable);
    expect(output.result).toMatchInlineSnapshot(
      `"<!doctype html><html><head><title>test</title><head><body><div><!--$-->Done<!-- --><!--/$--></div>"`,
    );
  });

  // @gate experimental
  it('should error the stream when an error is thrown at the root', async () => {
    const reportedErrors = [];
    const reportedShellErrors = [];
    const {writable, output, completed} = getTestWritable();
    const {pipe} = ReactDOMFizzServer.renderToPipeableStream(
      <div>
        <Throw />
      </div>,

      {
        onError(x) {
          reportedErrors.push(x);
        },
        onShellError(x) {
          reportedShellErrors.push(x);
        },
      },
    );

    // The stream is errored once we start writing.
    pipe(writable);

    await completed;

    expect(output.error).toBe(theError);
    expect(output.result).toBe('');
    // This type of error is reported to the error callback too.
    expect(reportedErrors).toEqual([theError]);
    expect(reportedShellErrors).toEqual([theError]);
  });

  // @gate experimental
  it('should error the stream when an error is thrown inside a fallback', async () => {
    const reportedErrors = [];
    const reportedShellErrors = [];
    const {writable, output, completed} = getTestWritable();
    const {pipe} = ReactDOMFizzServer.renderToPipeableStream(
      <div>
        <Suspense fallback={<Throw />}>
          <InfiniteSuspend />
        </Suspense>
      </div>,

      {
        onError(x) {
          reportedErrors.push(x);
        },
        onShellError(x) {
          reportedShellErrors.push(x);
        },
      },
    );
    pipe(writable);

    await completed;

    expect(output.error).toBe(theError);
    expect(output.result).toBe('');
    expect(reportedErrors).toEqual([theError]);
    expect(reportedShellErrors).toEqual([theError]);
  });

  // @gate experimental
  it('should not error the stream when an error is thrown inside suspense boundary', async () => {
    const reportedErrors = [];
    const reportedShellErrors = [];
    const {writable, output, completed} = getTestWritable();
    const {pipe} = ReactDOMFizzServer.renderToPipeableStream(
      <div>
        <Suspense fallback={<div>Loading</div>}>
          <Throw />
        </Suspense>
      </div>,

      {
        onError(x) {
          reportedErrors.push(x);
        },
        onShellError(x) {
          reportedShellErrors.push(x);
        },
      },
    );
    pipe(writable);

    await completed;

    expect(output.error).toBe(undefined);
    expect(output.result).toContain('Loading');
    // While no error is reported to the stream, the error is reported to the callback.
    expect(reportedErrors).toEqual([theError]);
    expect(reportedShellErrors).toEqual([]);
  });

  // @gate experimental
  it('should not attempt to render the fallback if the main content completes first', async () => {
    const {writable, output, completed} = getTestWritable();

    let renderedFallback = false;
    function Fallback() {
      renderedFallback = true;
      return 'Loading...';
    }
    function Content() {
      return 'Hi';
    }
    const {pipe} = ReactDOMFizzServer.renderToPipeableStream(
      <Suspense fallback={<Fallback />}>
        <Content />
      </Suspense>,
    );
    pipe(writable);

    await completed;

    expect(output.result).toContain('Hi');
    expect(output.result).not.toContain('Loading');
    expect(renderedFallback).toBe(false);
  });

  // @gate experimental
  it('should be able to complete by aborting even if the promise never resolves', async () => {
    let isCompleteCalls = 0;
    const {writable, output, completed} = getTestWritable();
    const {pipe, abort} = ReactDOMFizzServer.renderToPipeableStream(
      <div>
        <Suspense fallback={<div>Loading</div>}>
          <InfiniteSuspend />
        </Suspense>
      </div>,

      {
        onAllReady() {
          isCompleteCalls++;
        },
      },
    );
    pipe(writable);

    jest.runAllTimers();

    expect(output.result).toContain('Loading');
    expect(isCompleteCalls).toBe(0);

    abort();

    await completed;

    expect(output.error).toBe(undefined);
    expect(output.result).toContain('Loading');
    expect(isCompleteCalls).toBe(1);
  });

  // @gate experimental
  it('should be able to complete by abort when the fallback is also suspended', async () => {
    let isCompleteCalls = 0;
    const {writable, output, completed} = getTestWritable();
    const {pipe, abort} = ReactDOMFizzServer.renderToPipeableStream(
      <div>
        <Suspense fallback="Loading">
          <Suspense fallback={<InfiniteSuspend />}>
            <InfiniteSuspend />
          </Suspense>
        </Suspense>
      </div>,

      {
        onAllReady() {
          isCompleteCalls++;
        },
      },
    );
    pipe(writable);

    jest.runAllTimers();

    expect(output.result).toContain('Loading');
    expect(isCompleteCalls).toBe(0);

    abort();

    await completed;

    expect(output.error).toBe(undefined);
    expect(output.result).toContain('Loading');
    expect(isCompleteCalls).toBe(1);
  });

  // @gate experimental
  it('should be able to get context value when promise resolves', async () => {
    class DelayClient {
      get() {
        if (this.resolved) return this.resolved;
        if (this.pending) return this.pending;
        return (this.pending = new Promise(resolve => {
          setTimeout(() => {
            delete this.pending;
            this.resolved = 'OK';
            resolve();
          }, 500);
        }));
      }
    }

    const DelayContext = React.createContext(undefined);
    const Component = () => {
      const client = React.useContext(DelayContext);
      if (!client) {
        return 'context not found.';
      }
      const result = client.get();
      if (typeof result === 'string') {
        return result;
      }
      throw result;
    };

    const client = new DelayClient();
    const {writable, output, completed} = getTestWritable();
    ReactDOMFizzServer.renderToPipeableStream(
      <DelayContext.Provider value={client}>
        <Suspense fallback="loading">
          <Component />
        </Suspense>
      </DelayContext.Provider>,
    ).pipe(writable);

    jest.runAllTimers();

    expect(output.error).toBe(undefined);
    expect(output.result).toContain('loading');

    await completed;

    expect(output.error).toBe(undefined);
    expect(output.result).not.toContain('context never found');
    expect(output.result).toContain('OK');
  });

  // @gate experimental
  it('should be able to get context value when calls renderToPipeableStream twice at the same time', async () => {
    class DelayClient {
      get() {
        if (this.resolved) return this.resolved;
        if (this.pending) return this.pending;
        return (this.pending = new Promise(resolve => {
          setTimeout(() => {
            delete this.pending;
            this.resolved = 'OK';
            resolve();
          }, 500);
        }));
      }
    }
    const DelayContext = React.createContext(undefined);
    const Component = () => {
      const client = React.useContext(DelayContext);
      if (!client) {
        return 'context never found';
      }
      const result = client.get();
      if (typeof result === 'string') {
        return result;
      }
      throw result;
    };

    const client0 = new DelayClient();
    const {
      writable: writable0,
      output: output0,
      completed: completed0,
    } = getTestWritable();
    ReactDOMFizzServer.renderToPipeableStream(
      <DelayContext.Provider value={client0}>
        <Suspense fallback="loading">
          <Component />
        </Suspense>
      </DelayContext.Provider>,
    ).pipe(writable0);

    const client1 = new DelayClient();
    const {
      writable: writable1,
      output: output1,
      completed: completed1,
    } = getTestWritable();
    ReactDOMFizzServer.renderToPipeableStream(
      <DelayContext.Provider value={client1}>
        <Suspense fallback="loading">
          <Component />
        </Suspense>
      </DelayContext.Provider>,
    ).pipe(writable1);

    jest.runAllTimers();

    expect(output0.error).toBe(undefined);
    expect(output0.result).toContain('loading');

    expect(output1.error).toBe(undefined);
    expect(output1.result).toContain('loading');

    await Promise.all([completed0, completed1]);

    expect(output0.error).toBe(undefined);
    expect(output0.result).not.toContain('context never found');
    expect(output0.result).toContain('OK');

    expect(output1.error).toBe(undefined);
    expect(output1.result).not.toContain('context never found');
    expect(output1.result).toContain('OK');
  });

  // @gate experimental
  it('should be able to pop context after suspending', async () => {
    class DelayClient {
      get() {
        if (this.resolved) return this.resolved;
        if (this.pending) return this.pending;
        return (this.pending = new Promise(resolve => {
          setTimeout(() => {
            delete this.pending;
            this.resolved = 'OK';
            resolve();
          }, 500);
        }));
      }
    }

    const DelayContext = React.createContext(undefined);
    const Component = () => {
      const client = React.useContext(DelayContext);
      if (!client) {
        return 'context not found.';
      }
      const result = client.get();
      if (typeof result === 'string') {
        return result;
      }
      throw result;
    };

    const client = new DelayClient();
    const {writable, output, completed} = getTestWritable();
    ReactDOMFizzServer.renderToPipeableStream(
      <>
        <DelayContext.Provider value={client}>
          <Suspense fallback="loading">
            <Component />
          </Suspense>
        </DelayContext.Provider>
        <DelayContext.Provider value={client}>
          <Suspense fallback="loading">
            <Component />
          </Suspense>
        </DelayContext.Provider>
      </>,
    ).pipe(writable);

    jest.runAllTimers();

    expect(output.error).toBe(undefined);
    expect(output.result).toContain('loading');

    await completed;

    expect(output.error).toBe(undefined);
    expect(output.result).not.toContain('context never found');
    expect(output.result).toContain('OK');
  });

  // @gate experimental
  it('should not continue rendering after the writable ends unexpectedly', async () => {
    let hasLoaded = false;
    let resolve;
    let isComplete = false;
    let rendered = false;
    const promise = new Promise(r => (resolve = r));
    function Wait() {
      if (!hasLoaded) {
        throw promise;
      }
      rendered = true;
      return 'Done';
    }
    const {writable, completed} = getTestWritable();
    const {pipe} = ReactDOMFizzServer.renderToPipeableStream(
      <div>
        <Suspense fallback={<div>Loading</div>}>
          <Wait />
        </Suspense>
      </div>,
      {
        onAllReady() {
          isComplete = true;
        },
      },
    );
    pipe(writable);

    expect(rendered).toBe(false);
    expect(isComplete).toBe(false);

    writable.end();

    await jest.runAllTimers();

    hasLoaded = true;
    resolve();

    await completed;

    expect(rendered).toBe(false);
    expect(isComplete).toBe(true);
  });
});
