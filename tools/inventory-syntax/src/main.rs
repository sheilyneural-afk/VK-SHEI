use anyhow::{Context, Result, bail};
use serde_json::json;
use std::{env, fs, path::Path};
use tree_sitter::{Language, Node, Parser};

fn is_javascript_callable(node: Node<'_>) -> bool {
    matches!(
        node.kind(),
        "function_declaration"
            | "generator_function_declaration"
            | "method_definition"
            | "arrow_function"
            | "function_expression"
            | "generator_function"
    )
}

fn normalized_leaves(node: Node<'_>, source: &[u8], output: &mut String) {
    if node.kind().contains("comment") {
        return;
    }
    if node.child_count() == 0 {
        output.push_str(text(node, source));
        return;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        normalized_leaves(child, source, output);
    }
}

fn javascript_calls(root: Node<'_>, source: &[u8]) -> Vec<String> {
    fn visit(node: Node<'_>, root_id: usize, source: &[u8], calls: &mut Vec<String>) {
        if node.id() != root_id && is_javascript_callable(node) {
            return;
        }
        if node.kind() == "call_expression" {
            if let Some(function) = node.child_by_field_name("function") {
                let target = if function.kind() == "member_expression" {
                    function
                        .child_by_field_name("property")
                        .map(|item| text(item, source))
                } else if function.kind() == "identifier" {
                    Some(text(function, source))
                } else {
                    None
                };
                if let Some(target) = target.filter(|value| !value.is_empty()) {
                    calls.push(target.to_owned());
                }
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            visit(child, root_id, source, calls);
        }
    }
    let mut calls = Vec::new();
    visit(root, root.id(), source, &mut calls);
    calls.sort();
    calls.dedup();
    calls
}

fn bash_calls(root: Node<'_>, source: &[u8]) -> Vec<String> {
    fn visit(node: Node<'_>, root_id: usize, source: &[u8], calls: &mut Vec<String>) {
        if node.id() != root_id && node.kind() == "function_definition" {
            return;
        }
        if node.kind() == "command" {
            if let Some(name) = node.named_child(0) {
                if matches!(name.kind(), "command_name" | "word") {
                    let value = text(name, source);
                    if !value.is_empty() && !value.contains('$') {
                        calls.push(value.to_owned());
                    }
                }
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            visit(child, root_id, source, calls);
        }
    }
    let mut calls = Vec::new();
    visit(root, root.id(), source, &mut calls);
    calls.sort();
    calls.dedup();
    calls
}

fn point(node: Node<'_>) -> (usize, usize) {
    let point = node.start_position();
    (point.row + 1, point.column + 1)
}

fn text<'a>(node: Node<'_>, source: &'a [u8]) -> &'a str {
    node.utf8_text(source).unwrap_or("")
}

fn push_javascript(node: Node<'_>, source: &[u8], output: &mut Vec<serde_json::Value>) {
    let (kind, name_node) = match node.kind() {
        "function_declaration" | "generator_function_declaration" => {
            ("javascript_function", node.child_by_field_name("name"))
        }
        "method_definition" => ("javascript_method", node.child_by_field_name("name")),
        "variable_declarator" => {
            let value = node.child_by_field_name("value");
            if value.is_some_and(|value| {
                matches!(
                    value.kind(),
                    "arrow_function" | "function_expression" | "generator_function"
                )
            }) {
                (
                    "javascript_callable_binding",
                    node.child_by_field_name("name"),
                )
            } else {
                ("", None)
            }
        }
        _ => ("", None),
    };
    if let Some(name_node) = name_node {
        let (line, column) = point(name_node);
        let callable = if node.kind() == "variable_declarator" {
            node.child_by_field_name("value").unwrap_or(node)
        } else {
            node
        };
        let mut normalized_body = String::new();
        normalized_leaves(callable, source, &mut normalized_body);
        output.push(json!({
            "name": text(name_node, source),
            "line": line,
            "column": column,
            "end_line": node.end_position().row + 1,
            "kind": kind,
            "signature": text(node, source).split('{').next().unwrap_or("").trim(),
            "normalized_body": normalized_body,
            "call_targets": javascript_calls(callable, source),
        }));
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        push_javascript(child, source, output);
    }
}

fn push_bash(node: Node<'_>, source: &[u8], output: &mut Vec<serde_json::Value>) {
    if node.kind() == "function_definition" {
        if let Some(name_node) = node.child_by_field_name("name") {
            let (line, column) = point(name_node);
            let mut normalized_body = String::new();
            normalized_leaves(node, source, &mut normalized_body);
            output.push(json!({
                "name": text(name_node, source),
                "line": line,
                "column": column,
                "end_line": node.end_position().row + 1,
                "kind": "shell_function",
                "signature": text(node, source).split('{').next().unwrap_or("").trim(),
                "normalized_body": normalized_body,
                "call_targets": bash_calls(node, source),
            }));
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        push_bash(child, source, output);
    }
}

fn parse(language: Language, source: &[u8]) -> Result<tree_sitter::Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .context("configure tree-sitter language")?;
    let tree = parser
        .parse(source, None)
        .context("tree-sitter returned no tree")?;
    if tree.root_node().has_error() {
        bail!("syntax tree contains errors");
    }
    Ok(tree)
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let language = args
        .next()
        .context("usage: mslim-inventory-syntax <javascript|bash> <file>")?;
    let path = args.next().context("missing file")?;
    if args.next().is_some() {
        bail!("unexpected extra argument");
    }
    let source = fs::read(Path::new(&path)).with_context(|| format!("read {path}"))?;
    let mut output = Vec::new();
    match language.as_str() {
        "javascript" => {
            let tree = parse(tree_sitter_javascript::LANGUAGE.into(), &source)?;
            push_javascript(tree.root_node(), &source, &mut output);
        }
        "bash" => {
            let tree = parse(tree_sitter_bash::LANGUAGE.into(), &source)?;
            push_bash(tree.root_node(), &source, &mut output);
        }
        other => bail!("unsupported language: {other}"),
    }
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
