package com.atlantacrashrisk.backend.chatbot;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

@Service
public class ChatService {
	private static final String SYSTEM_INSTRUCTIONS = """
		You are the Atlanta crash-risk assistant. Use the area ranking tool for questions about
		which area has the highest predicted severity. Areas are 1 km² grid cells. Always state
		the shared scenario timestamp and explain that riskProbability is predicted serious/fatal
		severity conditional on a recorded crash, not the probability that a crash occurs.
		Do not invent a validated severity cutoff or combine model probabilities with historical
		severe-crash shares.
		""";

	private final ChatClient chatClient;

	public ChatService(ChatClient.Builder chatClientBuilder, AreaRankingTool areaRankingTool) {
		this.chatClient = chatClientBuilder.build();
		this.areaRankingTool = areaRankingTool;
	}

	private final AreaRankingTool areaRankingTool;

	public String respond(String userMessage) {
		return chatClient.prompt()
			.system(SYSTEM_INSTRUCTIONS)
			.tools(areaRankingTool)
			.user(userMessage)
			.call()
			.content();
	}
}
    