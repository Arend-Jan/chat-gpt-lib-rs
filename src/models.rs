use std::fmt::Result as FmtResult;
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// Define the OpenAIModel enum
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum Model {
    Gpt3_5Turbo,
}

// Implement Display to convert the enum back to a string
impl Display for Model {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let model_name = match self {
            Model::Gpt3_5Turbo => "gpt-3.5-turbo",
        };
        write!(f, "{}", model_name)
    }
}

// Implement `FromStr` to enable parsing the enum from a string
impl FromStr for Model {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gpt-3.5-turbo" => Ok(Model::Gpt3_5Turbo),
            _ => Err(()),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct LogitBias {
    pub biases: HashMap<u32, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_from_str() {
        let input = "gpt-3.5-turbo";
        let model: Result<Model, ()> = Model::from_str(input);
        assert!(model.is_ok(), "Failed to parse the model name");
        assert_eq!(model.unwrap(), Model::Gpt3_5Turbo);
    }

    #[test]
    fn test_from_str_invalid() {
        let input = "invalid-model";
        let model: Result<Model, ()> = Model::from_str(input);
        assert!(model.is_err(), "Parsed an invalid model name");
    }

    #[test]
    fn test_display() {
        let model = Model::Gpt3_5Turbo;
        let model_str = format!("{}", model);
        assert_eq!(model_str, "gpt-3.5-turbo");
    }

    #[test]
    fn test_serialize() {
        let model = Model::Gpt3_5Turbo;
        let serialized_model = serde_json::to_string(&model).unwrap();
        assert_eq!(serialized_model, "\"gpt-3.5-turbo\"");
    }

    #[test]
    fn test_deserialize() {
        let model_json = "\"gpt-3.5-turbo\"";
        let deserialized_model: Model = serde_json::from_str(model_json).unwrap();
        assert_eq!(deserialized_model, Model::Gpt3_5Turbo);
    }
}
