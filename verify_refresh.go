package main

import (
	"fmt"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"

	"lexiassist/services/user/internal/model"
)

func main() {
	dsn := "host=localhost user=lexiassist password=lexiassist_secret dbname=lexiassist port=5432 sslmode=disable"
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		fmt.Printf("FAILED: connect: %v\n", err)
		return
	}

	db.AutoMigrate(&model.RefreshToken{})

	create := &model.RefreshToken{
		TokenHash:  "testhash_" + fmt.Sprintf("%d", 1),
		DeviceInfo: model.DeviceInfo{DeviceName: "iPhone", OS: "iOS"},
		IPAddress:  "127.0.0.1",
	}
	if err := db.Create(create).Error; err != nil {
		fmt.Printf("FAILED: create: %v\n", err)
		return
	}
	fmt.Println("✓ Create succeeded")

	var read model.RefreshToken
	if err := db.First(&read, "token_hash = ?", create.TokenHash).Error; err != nil {
		fmt.Printf("FAILED: read: %v\n", err)
		return
	}
	fmt.Println("✓ Read succeeded")

	if read.DeviceInfo.DeviceName != "iPhone" {
		fmt.Printf("FAILED: data mismatch: %+v\n", read.DeviceInfo)
		return
	}
	fmt.Println("✓ Data verified")

	// Direct Scan test with string (simulates pgx)
	var di model.DeviceInfo
	if err := di.Scan(`{"device_name":"Pixel","os":"Android"}`); err != nil {
		fmt.Printf("FAILED: Scan(string): %v\n", err)
		return
	}
	fmt.Println("✓ Scan(string) succeeded")

	fmt.Println("\n=== ALL REFRESH TOKEN TESTS PASSED ===")
}
