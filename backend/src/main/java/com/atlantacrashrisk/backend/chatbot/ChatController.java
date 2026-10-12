package com.atlantacrashrisk.backend.chatbot;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;



@RestController
@RequestMapping("/chat")
public class ChatController {
    private final ChatService chatService;

    public ChatController(ChatService chatService) {
        this.chatService = chatService;
    }

    public static class ChatRequest {
        private String message;

        public String getMessage() {
            return message;
        }

        public void setMessage(String message) {
            this.message = message;
        }
    }

    @PostMapping
    public String handleChatRequest(@RequestBody ChatRequest chatRequest) {
        try {
            return chatService.respond(chatRequest.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            return "An error occurred while processing your request. Please Try again later.";
        }
    }
}