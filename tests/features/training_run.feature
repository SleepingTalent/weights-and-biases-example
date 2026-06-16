@e2e
Feature: Training run tracking
  As an ML engineer
  I want training experiments to be automatically tracked in W&B
  So that I can compare runs and reproduce results

  Scenario: A training run appears in the W&B dashboard
    Given the W&B server is running
    When I run a training experiment with 10 estimators
    Then a new run appears in the project dashboard
    And the run has train_accuracy metrics logged
    And a model artifact is attached to the run
