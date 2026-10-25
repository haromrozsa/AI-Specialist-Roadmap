package com.ai.specialist.llm;

import com.anthropic.client.AnthropicClient;
import com.anthropic.core.JsonValue;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.StopReason;
import com.anthropic.models.messages.TextBlock;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.util.stream.Collectors;

@RestController
public class ChatController {

    record ChatRequest(String prompt) {}

    record ChatResponse(String answer, String model, long inputTokens, long outputTokens) {}

    private final AnthropicClient client;
    private final String model;

    public ChatController(AnthropicClient client, @Value("${llm.model}") String model) {
        this.client = client;
        this.model = model;
    }

    @PostMapping("/api/chat")
    public ChatResponse chat(@RequestBody ChatRequest request) {
        if (request.prompt() == null || request.prompt().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "prompt must not be blank");
        }

        MessageCreateParams params = MessageCreateParams.builder()
                .model(model)
                .maxTokens(16000L)
                .system("You are a concise assistant. Answer in plain text.")
                .addUserMessage(request.prompt())
                // If a safety classifier declines, let the API retry on a fallback model
                .putAdditionalHeader("anthropic-beta", "server-side-fallback-2026-07-01")
                .putAdditionalBodyProperty("fallbacks", JsonValue.from("default"))
                .build();

        Message response = client.messages().create(params);

        if (response.stopReason().filter(StopReason.REFUSAL::equals).isPresent()) {
            throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY, "the model declined this request");
        }

        String answer = response.content().stream()
                .flatMap(block -> block.text().stream())
                .map(TextBlock::text)
                .collect(Collectors.joining());

        return new ChatResponse(answer, response.model().asString(),
                response.usage().inputTokens(), response.usage().outputTokens());
    }
}
