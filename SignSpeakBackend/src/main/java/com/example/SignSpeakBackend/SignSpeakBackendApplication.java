package com.example.SignSpeakBackend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class SignSpeakBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(SignSpeakBackendApplication.class, args);
	}

}
