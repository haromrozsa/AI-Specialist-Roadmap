package com.ai.specialist.llm;

import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class LlmApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(LlmApiApplication.class, args);
    }

    // One client for the whole app: it is thread-safe and holds the HTTP connection pool.
    // fromEnv() reads ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL, if set).
    @Bean(destroyMethod = "close")
    public AnthropicClient anthropicClient() {
        return AnthropicOkHttpClient.fromEnv();
    }
}
