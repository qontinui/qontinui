#!/usr/bin/env node

import { readFileSync } from 'fs';
import { parse as babelParse } from '@babel/parser';
import traverse from '@babel/traverse';
import ts from 'typescript';

/**
 * QontinUI TypeScript/JavaScript Parser
 *
 * Parses TypeScript and JavaScript files to extract:
 * - Component definitions (function, class, arrow function)
 * - React hooks (useState, useReducer, useContext, etc.)
 * - Conditional rendering patterns
 * - Event handlers
 * - Import/export relationships
 * - JSX element hierarchy
 */

class TypeScriptParser {
  constructor() {
    this.results = {};
  }

  /**
   * Parse a single file and extract all relevant information
   */
  parseFile(filePath, extractTypes) {
    try {
      const sourceCode = readFileSync(filePath, 'utf-8');
      const fileExt = filePath.split('.').pop();

      // Determine parser options based on file extension
      const parserOptions = {
        sourceType: 'module',
        plugins: [
          'jsx',
          'typescript',
          'classProperties',
          'decorators-legacy',
          'dynamicImport',
          'exportDefaultFrom',
          'exportNamespaceFrom',
          'objectRestSpread',
          'optionalChaining',
          'nullishCoalescingOperator',
        ],
      };

      let ast;
      try {
        ast = babelParse(sourceCode, parserOptions);
      } catch (parseError) {
        // If Babel fails, try with more permissive settings
        console.error(`Babel parse error for ${filePath}: ${parseError.message}`, { file: 'stderr' });
        return {
          error: parseError.message,
          components: [],
          state_variables: [],
          conditional_renders: [],
          event_handlers: [],
          imports: [],
          exports: [],
        };
      }

      const fileResult = {
        components: [],
        state_variables: [],
        conditional_renders: [],
        event_handlers: [],
        imports: [],
        exports: [],
        jsx_elements: [],
      };

      // Track scope for better context
      const scopeStack = [];
      let currentComponent = null;

      traverse.default(ast, {
        // Extract imports
        ImportDeclaration(path) {
          if (!extractTypes.includes('imports')) return;

          const importInfo = {
            source: path.node.source.value,
            specifiers: path.node.specifiers.map(spec => {
              if (spec.type === 'ImportDefaultSpecifier') {
                return { type: 'default', name: spec.local.name };
              } else if (spec.type === 'ImportNamespaceSpecifier') {
                return { type: 'namespace', name: spec.local.name };
              } else {
                return {
                  type: 'named',
                  name: spec.local.name,
                  imported: spec.imported.name,
                };
              }
            }),
            line: path.node.loc?.start.line || 0,
          };
          fileResult.imports.push(importInfo);
        },

        // Extract exports
        ExportNamedDeclaration(path) {
          if (!extractTypes.includes('imports')) return;

          if (path.node.declaration) {
            const { declaration } = path.node;
            if (declaration.type === 'VariableDeclaration') {
              declaration.declarations.forEach(decl => {
                fileResult.exports.push({
                  type: 'named',
                  name: decl.id.name,
                  line: path.node.loc?.start.line || 0,
                });
              });
            } else if (declaration.id) {
              fileResult.exports.push({
                type: 'named',
                name: declaration.id.name,
                line: path.node.loc?.start.line || 0,
              });
            }
          }
        },

        ExportDefaultDeclaration(path) {
          if (!extractTypes.includes('imports')) return;

          const name = path.node.declaration.id?.name ||
                      path.node.declaration.name ||
                      'default';
          fileResult.exports.push({
            type: 'default',
            name,
            line: path.node.loc?.start.line || 0,
          });
        },

        // Extract function components
        FunctionDeclaration(path) {
          if (!extractTypes.includes('components')) return;

          const component = this.extractFunctionComponent(path);
          if (component) {
            fileResult.components.push(component);
            currentComponent = component;
          }
        },

        ArrowFunctionExpression(path) {
          if (!extractTypes.includes('components')) return;

          // Check if this is assigned to a variable (component declaration)
          if (path.parent.type === 'VariableDeclarator') {
            const component = this.extractArrowComponent(path);
            if (component) {
              fileResult.components.push(component);
              currentComponent = component;
            }
          }
        },

        // Extract class components
        ClassDeclaration(path) {
          if (!extractTypes.includes('components')) return;

          const component = this.extractClassComponent(path);
          if (component) {
            fileResult.components.push(component);
            currentComponent = component;
          }
        },

        // Extract React hooks (useState, useReducer, useContext, etc.)
        CallExpression(path) {
          const calleeName = this.getCalleeName(path.node.callee);

          // Extract state hooks
          if (extractTypes.includes('state')) {
            if (calleeName === 'useState' || calleeName === 'useReducer' ||
                calleeName === 'useContext' || calleeName === 'useRef' ||
                calleeName === 'useMemo' || calleeName === 'useCallback') {
              const stateVar = this.extractHook(path, calleeName);
              if (stateVar) {
                fileResult.state_variables.push(stateVar);
              }
            }
          }

          // Extract event handlers
          if (extractTypes.includes('handlers')) {
            // Look for common event handler patterns
            if (path.parent.type === 'JSXExpressionContainer' &&
                path.parentPath.parent.type === 'JSXAttribute') {
              const attrName = path.parentPath.parent.name.name;
              if (attrName && attrName.startsWith('on')) {
                const handler = this.extractEventHandler(path, attrName);
                if (handler) {
                  fileResult.event_handlers.push(handler);
                }
              }
            }
          }
        },

        // Extract conditional rendering patterns
        JSXElement(path) {
          if (!extractTypes.includes('conditionals') && !extractTypes.includes('components')) return;

          // Extract JSX hierarchy for component analysis
          if (extractTypes.includes('components')) {
            const element = this.extractJSXElement(path);
            if (element) {
              fileResult.jsx_elements.push(element);
            }
          }
        },

        LogicalExpression(path) {
          if (!extractTypes.includes('conditionals')) return;

          // Detect {condition && <Component />} pattern
          if (path.node.operator === '&&' && this.isJSXExpression(path.node.right)) {
            const conditional = {
              condition: this.getConditionString(path.node.left),
              line: path.node.loc?.start.line || 0,
              pattern: 'AND',
              renders: this.getRenderedComponents(path.node.right),
            };
            fileResult.conditional_renders.push(conditional);
          }
        },

        ConditionalExpression(path) {
          if (!extractTypes.includes('conditionals')) return;

          // Detect ternary operator: condition ? <A /> : <B />
          if (this.isJSXExpression(path.node.consequent) ||
              this.isJSXExpression(path.node.alternate)) {
            const conditional = {
              condition: this.getConditionString(path.node.test),
              line: path.node.loc?.start.line || 0,
              pattern: 'TERNARY',
              renders_true: this.getRenderedComponents(path.node.consequent),
              renders_false: this.getRenderedComponents(path.node.alternate),
            };
            fileResult.conditional_renders.push(conditional);
          }
        },

        IfStatement(path) {
          if (!extractTypes.includes('conditionals')) return;

          // Detect early returns: if (condition) return <Component />
          if (path.node.consequent.type === 'ReturnStatement' &&
              this.isJSXExpression(path.node.consequent.argument)) {
            const conditional = {
              condition: this.getConditionString(path.node.test),
              line: path.node.loc?.start.line || 0,
              pattern: 'EARLY_RETURN',
              renders: this.getRenderedComponents(path.node.consequent.argument),
            };
            fileResult.conditional_renders.push(conditional);
          }
        },
      });

      return fileResult;
    } catch (error) {
      console.error(`Error parsing ${filePath}: ${error.message}`, { file: 'stderr' });
      return {
        error: error.message,
        components: [],
        state_variables: [],
        conditional_renders: [],
        event_handlers: [],
        imports: [],
        exports: [],
      };
    }
  }

  /**
   * Extract function component information
   */
  extractFunctionComponent(path) {
    const name = path.node.id?.name;
    if (!name) return null;

    // Check if it returns JSX
    let returnsJSX = false;
    path.traverse({
      ReturnStatement(returnPath) {
        if (this.isJSXExpression(returnPath.node.argument)) {
          returnsJSX = true;
          returnPath.stop();
        }
      },
    });

    if (!returnsJSX && !this.isCapitalized(name)) return null;

    const props = this.extractProps(path.node.params);
    const children = this.extractChildComponents(path);

    return {
      name,
      type: 'function',
      line: path.node.loc?.start.line || 0,
      props,
      children,
      returns_jsx: returnsJSX,
    };
  }

  /**
   * Extract arrow function component information
   */
  extractArrowComponent(path) {
    const name = path.parent.id?.name;
    if (!name) return null;

    let returnsJSX = false;

    // Check direct JSX return
    if (this.isJSXExpression(path.node.body)) {
      returnsJSX = true;
    } else {
      // Check for JSX in return statements
      path.traverse({
        ReturnStatement(returnPath) {
          if (this.isJSXExpression(returnPath.node.argument)) {
            returnsJSX = true;
            returnPath.stop();
          }
        },
      });
    }

    if (!returnsJSX && !this.isCapitalized(name)) return null;

    const props = this.extractProps(path.node.params);
    const children = this.extractChildComponents(path);

    return {
      name,
      type: 'arrow_function',
      line: path.node.loc?.start.line || 0,
      props,
      children,
      returns_jsx: returnsJSX,
    };
  }

  /**
   * Extract class component information
   */
  extractClassComponent(path) {
    const name = path.node.id?.name;
    if (!name) return null;

    // Check if it extends React.Component or Component
    const extendsReact = path.node.superClass && (
      path.node.superClass.name === 'Component' ||
      path.node.superClass.name === 'PureComponent' ||
      (path.node.superClass.object?.name === 'React' &&
       (path.node.superClass.property?.name === 'Component' ||
        path.node.superClass.property?.name === 'PureComponent'))
    );

    if (!extendsReact && !this.isCapitalized(name)) return null;

    const children = this.extractChildComponents(path);

    return {
      name,
      type: 'class',
      line: path.node.loc?.start.line || 0,
      props: [],
      children,
      extends: extendsReact ? 'React.Component' : 'unknown',
    };
  }

  /**
   * Extract hook information
   */
  extractHook(path, hookName) {
    const parent = path.parent;

    let varName = null;
    let setterName = null;
    let initialValue = null;

    // Handle destructured assignment: const [state, setState] = useState(...)
    if (parent.type === 'VariableDeclarator' && parent.id.type === 'ArrayPattern') {
      const elements = parent.id.elements;
      varName = elements[0]?.name || null;
      setterName = elements[1]?.name || null;
    } else if (parent.type === 'VariableDeclarator' && parent.id.type === 'Identifier') {
      // Handle non-destructured: const ref = useRef(...)
      varName = parent.id.name;
    }

    // Extract initial value
    if (path.node.arguments.length > 0) {
      initialValue = this.getValueString(path.node.arguments[0]);
    }

    return {
      name: varName,
      setter: setterName,
      hook: hookName,
      line: path.node.loc?.start.line || 0,
      initial_value: initialValue,
      type: this.inferType(initialValue),
    };
  }

  /**
   * Extract event handler information
   */
  extractEventHandler(path, eventName) {
    const handler = {
      event: eventName.replace(/^on/, '').toLowerCase(),
      line: path.node.loc?.start.line || 0,
      state_changes: [],
    };

    // Try to get handler name
    if (path.node.type === 'Identifier') {
      handler.name = path.node.name;
    } else if (path.node.type === 'ArrowFunctionExpression' ||
               path.node.type === 'FunctionExpression') {
      handler.name = 'inline';

      // Look for state setter calls inside the handler
      path.traverse({
        CallExpression(callPath) {
          const calleeName = this.getCalleeName(callPath.node.callee);
          if (calleeName && calleeName.startsWith('set')) {
            handler.state_changes.push(calleeName);
          }
        },
      });
    } else if (path.node.type === 'CallExpression') {
      handler.name = this.getCalleeName(path.node.callee);
    }

    return handler;
  }

  /**
   * Extract JSX element information
   */
  extractJSXElement(path) {
    const openingElement = path.node.openingElement;
    const name = this.getJSXElementName(openingElement.name);

    return {
      name,
      line: path.node.loc?.start.line || 0,
      props: openingElement.attributes.map(attr => {
        if (attr.type === 'JSXAttribute') {
          return {
            name: attr.name.name,
            value: attr.value ? this.getValueString(attr.value) : true,
          };
        }
        return null;
      }).filter(Boolean),
      self_closing: path.node.openingElement.selfClosing,
    };
  }

  /**
   * Extract component props from function parameters
   */
  extractProps(params) {
    if (!params || params.length === 0) return [];

    const firstParam = params[0];
    const props = [];

    if (firstParam.type === 'ObjectPattern') {
      firstParam.properties.forEach(prop => {
        if (prop.type === 'ObjectProperty') {
          props.push({
            name: prop.key.name,
            default: prop.value.type === 'AssignmentPattern' ?
                     this.getValueString(prop.value.right) : undefined,
          });
        }
      });
    } else if (firstParam.type === 'Identifier') {
      props.push({ name: firstParam.name });
    }

    return props;
  }

  /**
   * Extract child component names from JSX
   */
  extractChildComponents(path) {
    const children = new Set();

    path.traverse({
      JSXElement(jsxPath) {
        const name = this.getJSXElementName(jsxPath.node.openingElement.name);
        if (name && this.isCapitalized(name)) {
          children.add(name);
        }
      },
    });

    return Array.from(children);
  }

  /**
   * Get JSX element name
   */
  getJSXElementName(node) {
    if (!node) return null;

    if (node.type === 'JSXIdentifier') {
      return node.name;
    } else if (node.type === 'JSXMemberExpression') {
      return `${this.getJSXElementName(node.object)}.${node.property.name}`;
    }
    return null;
  }

  /**
   * Get callee name from CallExpression
   */
  getCalleeName(node) {
    if (!node) return null;

    if (node.type === 'Identifier') {
      return node.name;
    } else if (node.type === 'MemberExpression') {
      return node.property.name;
    }
    return null;
  }

  /**
   * Check if expression is JSX
   */
  isJSXExpression(node) {
    if (!node) return false;
    return node.type === 'JSXElement' || node.type === 'JSXFragment';
  }

  /**
   * Get condition as string
   */
  getConditionString(node) {
    if (!node) return '';

    if (node.type === 'Identifier') {
      return node.name;
    } else if (node.type === 'MemberExpression') {
      return `${this.getConditionString(node.object)}.${node.property.name}`;
    } else if (node.type === 'UnaryExpression') {
      return `${node.operator}${this.getConditionString(node.argument)}`;
    } else if (node.type === 'BinaryExpression' || node.type === 'LogicalExpression') {
      return `${this.getConditionString(node.left)} ${node.operator} ${this.getConditionString(node.right)}`;
    }

    return 'complex_condition';
  }

  /**
   * Get rendered components from JSX expression
   */
  getRenderedComponents(node) {
    if (!node) return [];

    const components = [];

    if (node.type === 'JSXElement') {
      const name = this.getJSXElementName(node.openingElement.name);
      if (name) components.push(name);
    } else if (node.type === 'JSXFragment') {
      components.push('Fragment');
    }

    return components;
  }

  /**
   * Get value as string
   */
  getValueString(node) {
    if (!node) return null;

    if (node.type === 'StringLiteral') {
      return node.value;
    } else if (node.type === 'NumericLiteral') {
      return String(node.value);
    } else if (node.type === 'BooleanLiteral') {
      return String(node.value);
    } else if (node.type === 'NullLiteral') {
      return 'null';
    } else if (node.type === 'Identifier') {
      return node.name;
    } else if (node.type === 'ArrayExpression') {
      return '[]';
    } else if (node.type === 'ObjectExpression') {
      return '{}';
    } else if (node.type === 'JSXExpressionContainer') {
      return this.getValueString(node.expression);
    }

    return 'unknown';
  }

  /**
   * Infer type from initial value
   */
  inferType(value) {
    if (value === null) return 'unknown';
    if (value === 'true' || value === 'false') return 'boolean';
    if (value === '[]') return 'array';
    if (value === '{}') return 'object';
    if (value === 'null') return 'null';
    if (!isNaN(Number(value))) return 'number';
    return 'string';
  }

  /**
   * Check if name is capitalized (likely a component)
   */
  isCapitalized(name) {
    return name && name[0] === name[0].toUpperCase();
  }

  /**
   * Main parse function
   */
  parse(config) {
    const { files, extract } = config;
    const results = {};

    for (const file of files) {
      results[file] = this.parseFile(file, extract);
    }

    return { files: results };
  }
}

/**
 * Main execution
 */
async function main() {
  try {
    // Read configuration from stdin
    const chunks = [];
    for await (const chunk of process.stdin) {
      chunks.push(chunk);
    }
    const input = Buffer.concat(chunks).toString('utf-8');

    if (!input.trim()) {
      throw new Error('No input configuration provided');
    }

    const config = JSON.parse(input);

    if (!config.files || !Array.isArray(config.files)) {
      throw new Error('Invalid configuration: "files" array is required');
    }

    if (!config.extract || !Array.isArray(config.extract)) {
      config.extract = ['components', 'state', 'conditionals', 'handlers', 'imports'];
    }

    const parser = new TypeScriptParser();
    const results = parser.parse(config);

    // Output results to stdout
    console.log(JSON.stringify(results, null, 2));
    process.exit(0);
  } catch (error) {
    console.error(`Fatal error: ${error.message}`, { file: 'stderr' });
    process.exit(1);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { TypeScriptParser };
