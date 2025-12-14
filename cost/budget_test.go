package cost

import (
	"fmt"
	"strings"
	"sync"
	"testing"
)

func TestNewBudgetManager(t *testing.T) {
	tests := []struct {
		name      string
		maxBudget float64
	}{
		{
			name:      "with budget limit",
			maxBudget: 10.0,
		},
		{
			name:      "no budget limit",
			maxBudget: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bm := NewBudgetManager(tt.maxBudget)
			if bm == nil {
				t.Fatal("NewBudgetManager returned nil")
			}
			if bm.maxBudget != tt.maxBudget {
				t.Errorf("maxBudget = %f, want %f", bm.maxBudget, tt.maxBudget)
			}
			if bm.currentCost != 0 {
				t.Errorf("currentCost = %f, want 0", bm.currentCost)
			}
		})
	}
}

func TestBudgetManager_UpdateCost(t *testing.T) {
	tests := []struct {
		name      string
		maxBudget float64
		updates   []struct {
			cost    float64
			model   string
			user    string
			wantErr bool
		}
		wantTotal float64
	}{
		{
			name:      "single update within budget",
			maxBudget: 10.0,
			updates: []struct {
				cost    float64
				model   string
				user    string
				wantErr bool
			}{
				{cost: 5.0, model: "gpt-4", user: "user1", wantErr: false},
			},
			wantTotal: 5.0,
		},
		{
			name:      "multiple updates within budget",
			maxBudget: 10.0,
			updates: []struct {
				cost    float64
				model   string
				user    string
				wantErr bool
			}{
				{cost: 3.0, model: "gpt-4", user: "user1", wantErr: false},
				{cost: 2.0, model: "gpt-3.5-turbo", user: "user2", wantErr: false},
				{cost: 1.5, model: "gpt-4", user: "user1", wantErr: false},
			},
			wantTotal: 6.5,
		},
		{
			name:      "exceed budget",
			maxBudget: 5.0,
			updates: []struct {
				cost    float64
				model   string
				user    string
				wantErr bool
			}{
				{cost: 3.0, model: "gpt-4", user: "user1", wantErr: false},
				{cost: 3.0, model: "gpt-4", user: "user2", wantErr: true}, // Should fail
			},
			wantTotal: 3.0, // Only first update should succeed
		},
		{
			name:      "no budget limit",
			maxBudget: 0,
			updates: []struct {
				cost    float64
				model   string
				user    string
				wantErr bool
			}{
				{cost: 100.0, model: "gpt-4", user: "user1", wantErr: false},
				{cost: 200.0, model: "gpt-4", user: "user2", wantErr: false},
			},
			wantTotal: 300.0,
		},
		{
			name:      "update with empty user",
			maxBudget: 10.0,
			updates: []struct {
				cost    float64
				model   string
				user    string
				wantErr bool
			}{
				{cost: 2.0, model: "gpt-4", user: "", wantErr: false},
			},
			wantTotal: 2.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bm := NewBudgetManager(tt.maxBudget)

			for i, update := range tt.updates {
				err := bm.UpdateCost(update.cost, update.model, update.user)

				if (err != nil) != update.wantErr {
					t.Errorf("Update %d: UpdateCost() error = %v, wantErr %v", i, err, update.wantErr)
				}

				// Verify error message contains budget information
				if update.wantErr && err != nil {
					errMsg := err.Error()
					if !strings.Contains(errMsg, "budget exceeded") {
						t.Errorf("Update %d: Expected 'budget exceeded' in error, got: %s", i, errMsg)
					}
				}
			}

			// Verify total cost
			total := bm.GetCurrentCost()
			if total != tt.wantTotal {
				t.Errorf("GetCurrentCost() = %f, want %f", total, tt.wantTotal)
			}
		})
	}
}

func TestBudgetManager_GetCostByModel(t *testing.T) {
	bm := NewBudgetManager(100.0)

	// Add costs for different models
	updates := []struct {
		cost  float64
		model string
		user  string
	}{
		{cost: 5.0, model: "gpt-4", user: "user1"},
		{cost: 3.0, model: "gpt-3.5-turbo", user: "user1"},
		{cost: 2.5, model: "gpt-4", user: "user2"},
		{cost: 1.5, model: "gpt-3.5-turbo", user: "user2"},
	}

	for _, update := range updates {
		if err := bm.UpdateCost(update.cost, update.model, update.user); err != nil {
			t.Fatalf("UpdateCost failed: %v", err)
		}
	}

	costsByModel := bm.GetCostByModel()

	expected := map[string]float64{
		"gpt-4":         7.5, // 5.0 + 2.5
		"gpt-3.5-turbo": 4.5, // 3.0 + 1.5
	}

	for model, wantCost := range expected {
		gotCost, exists := costsByModel[model]
		if !exists {
			t.Errorf("Model %s not found in cost breakdown", model)
			continue
		}
		if gotCost != wantCost {
			t.Errorf("Model %s: cost = %f, want %f", model, gotCost, wantCost)
		}
	}

	// Verify map is a copy (not a reference)
	costsByModel["gpt-4"] = 999.0
	if bm.GetCostByModel()["gpt-4"] == 999.0 {
		t.Error("GetCostByModel returned reference to internal map instead of copy")
	}
}

func TestBudgetManager_GetCostByUser(t *testing.T) {
	bm := NewBudgetManager(100.0)

	// Add costs for different users
	updates := []struct {
		cost  float64
		model string
		user  string
	}{
		{cost: 5.0, model: "gpt-4", user: "alice"},
		{cost: 3.0, model: "gpt-3.5-turbo", user: "bob"},
		{cost: 2.5, model: "gpt-4", user: "alice"},
		{cost: 1.5, model: "gpt-3.5-turbo", user: "bob"},
		{cost: 4.0, model: "gpt-4", user: ""}, // Empty user
	}

	for _, update := range updates {
		if err := bm.UpdateCost(update.cost, update.model, update.user); err != nil {
			t.Fatalf("UpdateCost failed: %v", err)
		}
	}

	costsByUser := bm.GetCostByUser()

	expected := map[string]float64{
		"alice": 7.5, // 5.0 + 2.5
		"bob":   4.5, // 3.0 + 1.5
		// Empty user should not be tracked
	}

	for user, wantCost := range expected {
		gotCost, exists := costsByUser[user]
		if !exists {
			t.Errorf("User %s not found in cost breakdown", user)
			continue
		}
		if gotCost != wantCost {
			t.Errorf("User %s: cost = %f, want %f", user, gotCost, wantCost)
		}
	}

	// Verify empty user was not tracked
	if _, exists := costsByUser[""]; exists {
		t.Error("Empty user should not be tracked in cost breakdown")
	}

	// Verify map is a copy (not a reference)
	costsByUser["alice"] = 999.0
	if bm.GetCostByUser()["alice"] == 999.0 {
		t.Error("GetCostByUser returned reference to internal map instead of copy")
	}
}

func TestBudgetManager_Reset(t *testing.T) {
	bm := NewBudgetManager(10.0)

	// Add some costs
	if err := bm.UpdateCost(3.0, "gpt-4", "user1"); err != nil {
		t.Fatalf("UpdateCost failed: %v", err)
	}
	if err := bm.UpdateCost(2.0, "gpt-3.5-turbo", "user2"); err != nil {
		t.Fatalf("UpdateCost failed: %v", err)
	}

	// Verify costs were added
	if bm.GetCurrentCost() != 5.0 {
		t.Errorf("Before reset: currentCost = %f, want 5.0", bm.GetCurrentCost())
	}

	// Reset
	bm.Reset()

	// Verify everything was reset
	if bm.GetCurrentCost() != 0 {
		t.Errorf("After reset: currentCost = %f, want 0", bm.GetCurrentCost())
	}

	costsByModel := bm.GetCostByModel()
	if len(costsByModel) != 0 {
		t.Errorf("After reset: costsByModel not empty, got %v", costsByModel)
	}

	costsByUser := bm.GetCostByUser()
	if len(costsByUser) != 0 {
		t.Errorf("After reset: costsByUser not empty, got %v", costsByUser)
	}

	// Verify we can add costs again after reset
	if err := bm.UpdateCost(1.0, "gpt-4", "user1"); err != nil {
		t.Errorf("UpdateCost after reset failed: %v", err)
	}
	if bm.GetCurrentCost() != 1.0 {
		t.Errorf("After reset and new update: currentCost = %f, want 1.0", bm.GetCurrentCost())
	}
}

func TestBudgetManager_BudgetEnforcement(t *testing.T) {
	tests := []struct {
		name      string
		maxBudget float64
		cost1     float64
		cost2     float64
		wantErr2  bool
	}{
		{
			name:      "exactly at budget limit",
			maxBudget: 10.0,
			cost1:     5.0,
			cost2:     5.0,
			wantErr2:  false,
		},
		{
			name:      "one cent over budget",
			maxBudget: 10.0,
			cost1:     5.0,
			cost2:     5.01,
			wantErr2:  true,
		},
		{
			name:      "way over budget",
			maxBudget: 10.0,
			cost1:     5.0,
			cost2:     100.0,
			wantErr2:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bm := NewBudgetManager(tt.maxBudget)

			// First update should always succeed
			if err := bm.UpdateCost(tt.cost1, "gpt-4", "user1"); err != nil {
				t.Fatalf("First UpdateCost failed: %v", err)
			}

			// Second update may fail
			err := bm.UpdateCost(tt.cost2, "gpt-4", "user2")
			if (err != nil) != tt.wantErr2 {
				t.Errorf("Second UpdateCost() error = %v, wantErr %v", err, tt.wantErr2)
			}

			// Verify cost was not updated if error occurred
			expectedCost := tt.cost1
			if !tt.wantErr2 {
				expectedCost += tt.cost2
			}

			if bm.GetCurrentCost() != expectedCost {
				t.Errorf("currentCost = %f, want %f", bm.GetCurrentCost(), expectedCost)
			}
		})
	}
}

func TestBudgetManager_Concurrency(t *testing.T) {
	bm := NewBudgetManager(0) // No limit for concurrency test

	var wg sync.WaitGroup
	iterations := 100
	costPerUpdate := 0.01

	// Concurrent updates
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			model := fmt.Sprintf("model-%d", idx%5)
			user := fmt.Sprintf("user-%d", idx%10)
			if err := bm.UpdateCost(costPerUpdate, model, user); err != nil {
				t.Errorf("UpdateCost failed: %v", err)
			}
		}(i)
	}

	// Concurrent reads
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = bm.GetCurrentCost()
			_ = bm.GetCostByModel()
			_ = bm.GetCostByUser()
		}()
	}

	wg.Wait()

	// Verify final cost
	expectedTotal := float64(iterations) * costPerUpdate
	actualTotal := bm.GetCurrentCost()

	// Use tolerance for floating point comparison
	tolerance := 0.0001
	if actualTotal < expectedTotal-tolerance || actualTotal > expectedTotal+tolerance {
		t.Errorf("After concurrent updates: currentCost = %f, want %f", actualTotal, expectedTotal)
	}

	// Verify cost breakdowns
	costsByModel := bm.GetCostByModel()
	costsByUser := bm.GetCostByUser()

	if len(costsByModel) != 5 {
		t.Errorf("Expected 5 models, got %d", len(costsByModel))
	}

	if len(costsByUser) != 10 {
		t.Errorf("Expected 10 users, got %d", len(costsByUser))
	}
}

func TestBudgetManager_ConcurrentBudgetEnforcement(t *testing.T) {
	maxBudget := 1.0
	bm := NewBudgetManager(maxBudget)

	var wg sync.WaitGroup
	attempts := 100
	costPerUpdate := 0.05

	successCount := 0
	var mu sync.Mutex

	// Try to exceed budget concurrently
	for i := 0; i < attempts; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			err := bm.UpdateCost(costPerUpdate, "gpt-4", fmt.Sprintf("user-%d", idx))
			if err == nil {
				mu.Lock()
				successCount++
				mu.Unlock()
			}
		}(i)
	}

	wg.Wait()

	// Verify budget was not exceeded
	totalCost := bm.GetCurrentCost()
	if totalCost > maxBudget {
		t.Errorf("Budget exceeded: totalCost = %f, maxBudget = %f", totalCost, maxBudget)
	}

	// Verify at least some updates succeeded
	if successCount == 0 {
		t.Error("No updates succeeded, expected some to succeed before hitting budget")
	}

	// Verify success count matches total cost
	expectedSuccesses := int(totalCost / costPerUpdate)
	tolerance := 2 // Allow small variance due to concurrency

	if successCount < expectedSuccesses-tolerance || successCount > expectedSuccesses+tolerance {
		t.Errorf("Success count = %d, expected around %d (tolerance ±%d)",
			successCount, expectedSuccesses, tolerance)
	}
}
