"""
Mini AI Model Trainer Framework
Demonstrates: class attributes, instance attributes, abstraction (ABC),
single inheritance, method overriding, super(), polymorphism,
composition, aggregation, magic methods, and instance methods.
"""

from abc import ABC, abstractmethod


# ─────────────────────────────────────────────
# 1. ModelConfig  (Composition object)
# ─────────────────────────────────────────────
class ModelConfig:
    """Stores model hyper-parameters. Used via composition inside BaseModel."""

    def __init__(self, model_name: str, learning_rate: float = 0.01, epochs: int = 10):
        # Instance attributes
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Magic method — __repr__
    def __repr__(self) -> str:
        return (
            f"[Config] {self.model_name:<15} | "
            f"lr={self.learning_rate} | "
            f"epochs={self.epochs}"
        )


# ─────────────────────────────────────────────
# 2. BaseModel  (Abstract Base Class)
# ─────────────────────────────────────────────
class BaseModel(ABC):
    """
    Abstract base class for all models.
    Enforces a common interface via @abstractmethod.
    Tracks total instances via a class attribute.
    """

    # Class attribute shared across all instances
    model_count: int = 0

    def __init__(self, config: ModelConfig):
        # Instance attribute — composition: BaseModel *owns* a ModelConfig
        self.config = config
        BaseModel.model_count += 1  # update class attribute on every instantiation

    @abstractmethod
    def train(self, data) -> None:
        """Train the model on the supplied data."""
        ...

    @abstractmethod
    def evaluate(self, data) -> None:
        """Evaluate the model on the supplied data."""
        ...


# ─────────────────────────────────────────────
# 3. LinearRegressionModel  (Concrete, inherits BaseModel)
# ─────────────────────────────────────────────
class LinearRegressionModel(BaseModel):
    """
    Single inheritance: LinearRegressionModel → BaseModel
    Overrides train() and evaluate().
    Calls parent __init__ via super().
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)          # super() — delegates to BaseModel.__init__

    # Method overriding
    def train(self, data) -> None:
        n_samples = len(data)
        print(
            f"LinearRegression: Training on {n_samples} samples "
            f"for {self.config.epochs} epochs "
            f"(lr={self.config.learning_rate})"
        )

    # Method overriding
    def evaluate(self, data) -> None:
        # Simulated MSE result
        print("LinearRegression: Evaluation MSE = 0.042")


# ─────────────────────────────────────────────
# 4. NeuralNetworkModel  (Concrete, inherits BaseModel)
# ─────────────────────────────────────────────
class NeuralNetworkModel(BaseModel):
    """
    Single inheritance: NeuralNetworkModel → BaseModel
    Adds an extra instance attribute `layers`.
    Overrides train() and evaluate() differently (polymorphism).
    """

    def __init__(self, config: ModelConfig, layers: list[int] = None):
        super().__init__(config)          # super() — delegates to BaseModel.__init__
        # Extra instance attribute specific to this subclass
        self.layers: list[int] = layers if layers is not None else [64, 32, 1]

    # Method overriding
    def train(self, data) -> None:
        n_samples = len(data)
        print(
            f"NeuralNetwork {self.layers}: Training on {n_samples} samples "
            f"for {self.config.epochs} epochs "
            f"(lr={self.config.learning_rate})"
        )

    # Method overriding
    def evaluate(self, data) -> None:
        # Simulated accuracy result
        print("NeuralNetwork: Evaluation Accuracy = 91.5%")


# ─────────────────────────────────────────────
# 5. DataLoader  (Independent class — used via aggregation)
# ─────────────────────────────────────────────
class DataLoader:
    """
    Holds a dataset independently of any model.
    Passed into Trainer as aggregation (Trainer does NOT own DataLoader).
    """

    def __init__(self, dataset: list):
        self.dataset = dataset          # instance attribute

    def load(self) -> list:
        """Return the dataset (instance method)."""
        return self.dataset


# ─────────────────────────────────────────────
# 6. Trainer  (Orchestrator — aggregation of DataLoader)
# ─────────────────────────────────────────────
class Trainer:
    """
    Accepts any BaseModel + DataLoader (aggregation — does not own them).
    run() demonstrates polymorphism: works with ANY BaseModel subclass.
    """

    def __init__(self, model: BaseModel, data_loader: DataLoader):
        # Aggregation: Trainer receives these objects externally
        self.model = model
        self.data_loader = data_loader

    def run(self) -> None:
        """Orchestrate: load data → train → evaluate (instance method)."""
        data = self.data_loader.load()
        model_name = self.model.config.model_name
        print(f"\n--- Training {model_name} ---")
        self.model.train(data)      # polymorphic call — dispatches to the correct subclass
        self.model.evaluate(data)   # polymorphic call


# ─────────────────────────────────────────────
# 7. Main — wire everything together
# ─────────────────────────────────────────────
def main():
    # --- Configs (composition objects) ---
    lr_config = ModelConfig("LinearRegression", learning_rate=0.01, epochs=10)
    nn_config = ModelConfig("NeuralNetwork",    learning_rate=0.001, epochs=20)

    # __repr__ magic method in action
    print(lr_config)
    print(nn_config)

    # --- Models (class attribute model_count updated on each instantiation) ---
    lr_model = LinearRegressionModel(config=lr_config)
    nn_model = NeuralNetworkModel(config=nn_config, layers=[64, 32, 1])

    print(f"Models created: {BaseModel.model_count}")

    # --- Shared DataLoader (aggregation) ---
    data_loader = DataLoader(dataset=[1, 2, 3, 4, 5])

    # --- Trainers (polymorphism: same Trainer.run() works for both models) ---
    trainer_lr = Trainer(model=lr_model, data_loader=data_loader)
    trainer_nn = Trainer(model=nn_model, data_loader=data_loader)

    trainer_lr.run()
    trainer_nn.run()


if __name__ == "__main__":
    main()