package cost

import (
	"fmt"
	"sync"
)

// BudgetManager manages cost tracking and budget limits.
//
// Thread Safety: BudgetManager is safe for concurrent use.
type BudgetManager struct {
	maxBudget   float64
	currentCost float64
	costByModel map[string]float64
	costByUser  map[string]float64
	mu          sync.RWMutex
}

// NewBudgetManager creates a new budget manager.
//
// maxBudget of 0 means no limit.
func NewBudgetManager(maxBudget float64) *BudgetManager {
	return &BudgetManager{
		maxBudget:   maxBudget,
		costByModel: make(map[string]float64),
		costByUser:  make(map[string]float64),
	}
}

// UpdateCost updates the current cost.
//
// Returns error if budget would be exceeded.
func (b *BudgetManager) UpdateCost(cost float64, model, user string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	newCost := b.currentCost + cost

	// Check budget before updating (prevent exceeding)
	if b.maxBudget > 0 && newCost > b.maxBudget {
		return fmt.Errorf("budget exceeded: current=$%.4f, max=$%.4f, attempted=$%.4f",
			b.currentCost, b.maxBudget, cost)
	}

	// Update costs
	b.currentCost = newCost
	b.costByModel[model] += cost
	if user != "" {
		b.costByUser[user] += cost
	}

	return nil
}

// GetCurrentCost returns the current total cost.
func (b *BudgetManager) GetCurrentCost() float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.currentCost
}

// GetCostByModel returns cost breakdown by model.
func (b *BudgetManager) GetCostByModel() map[string]float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()

	costs := make(map[string]float64)
	for k, v := range b.costByModel {
		costs[k] = v
	}
	return costs
}

// GetCostByUser returns cost breakdown by user.
func (b *BudgetManager) GetCostByUser() map[string]float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()

	costs := make(map[string]float64)
	for k, v := range b.costByUser {
		costs[k] = v
	}
	return costs
}

// Reset resets all cost tracking.
func (b *BudgetManager) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.currentCost = 0
	b.costByModel = make(map[string]float64)
	b.costByUser = make(map[string]float64)
}
