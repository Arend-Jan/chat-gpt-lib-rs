use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Function,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    StringChoice(String),
    ObjectChoice(ToolChoiceObject),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolChoiceObject {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn test_tool_serialization() {
        let function = Function {
            name: "test_function".to_string(),
            description: Some("A test function".to_string()),
            parameters: Some(json!({"param1": "value1"})),
        };

        let tool = Tool {
            tool_type: "test_tool".to_string(),
            function,
        };

        let expected_json = json!({
            "type": "test_tool",
            "function": {
                "name": "test_function",
                "description": "A test function",
                "parameters": {
                    "param1": "value1"
                }
            }
        });

        let serialized = serde_json::to_string(&tool).unwrap();
        let deserialized: Value = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, expected_json);
    }

    #[test]
    fn test_tool_deserialization() {
        let json_data = json!({
            "type": "test_tool",
            "function": {
                "name": "test_function",
                "description": "A test function",
                "parameters": {
                    "param1": "value1"
                }
            }
        })
        .to_string();

        let expected_tool = Tool {
            tool_type: "test_tool".to_string(),
            function: Function {
                name: "test_function".to_string(),
                description: Some("A test function".to_string()),
                parameters: Some(json!({"param1": "value1"})),
            },
        };

        let deserialized: Tool = serde_json::from_str(&json_data).unwrap();

        assert_eq!(deserialized.tool_type, expected_tool.tool_type);
        assert_eq!(deserialized.function.name, expected_tool.function.name);
        assert_eq!(
            deserialized.function.description,
            expected_tool.function.description
        );
        assert_eq!(
            deserialized.function.parameters,
            expected_tool.function.parameters
        );
    }

    #[test]
    fn test_tool_choice_serialization() {
        let tool_choice_str = ToolChoice::StringChoice("test_choice".to_string());
        let serialized_str = serde_json::to_string(&tool_choice_str).unwrap();
        let deserialized_str: ToolChoice = serde_json::from_str(&serialized_str).unwrap();

        match deserialized_str {
            ToolChoice::StringChoice(value) => assert_eq!(value, "test_choice"),
            _ => panic!("Expected StringChoice"),
        }

        let tool_choice_obj = ToolChoice::ObjectChoice(ToolChoiceObject {
            tool_type: "test_tool".to_string(),
            function: ToolFunction {
                name: "test_function".to_string(),
            },
        });

        let expected_json = json!({
            "type": "test_tool",
            "function": {
                "name": "test_function",
            }
        });

        let serialized_obj = serde_json::to_string(&tool_choice_obj).unwrap();
        let deserialized_obj: Value = serde_json::from_str(&serialized_obj).unwrap();

        assert_eq!(deserialized_obj, expected_json);
    }

    #[test]
    fn test_tool_choice_deserialization() {
        let json_str = "\"test_choice\"";
        let deserialized_str: ToolChoice = serde_json::from_str(json_str).unwrap();

        match deserialized_str {
            ToolChoice::StringChoice(value) => assert_eq!(value, "test_choice"),
            _ => panic!("Expected StringChoice"),
        }

        let json_obj = json!({
            "type": "test_tool",
            "function": {
                "name": "test_function",
            }
        })
        .to_string();

        let deserialized_obj: ToolChoice = serde_json::from_str(&json_obj).unwrap();

        match deserialized_obj {
            ToolChoice::ObjectChoice(obj) => {
                assert_eq!(obj.tool_type, "test_tool");
                assert_eq!(obj.function.name, "test_function");
            }
            _ => panic!("Expected ObjectChoice"),
        }
    }
}
