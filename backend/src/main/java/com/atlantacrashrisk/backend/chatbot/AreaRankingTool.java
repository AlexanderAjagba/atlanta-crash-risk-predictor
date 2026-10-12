package com.atlantacrashrisk.backend.chatbot;

import java.time.format.DateTimeFormatter;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.stereotype.Component;

@Component
public class AreaRankingTool {
    private static final DateTimeFormatter SCENARIO_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private final AreaRankingClient client;

    public AreaRankingTool(AreaRankingClient client) {
        this.client = client;
    }

    @Tool(description = "Rank Atlanta 1 km² grid cells by predicted serious/fatal crash severity. Use this for questions asking which area has the highest predicted severity. Use one shared Atlanta-local scenario timestamp for all cells. The result is conditional on a recorded crash and is not crash occurrence probability.")
    public AreaRankingClient.AreaRankingResponse rankHighestSeverityArea() {
        String scenario = ZonedDateTime.now(ZoneId.of("America/New_York")).format(SCENARIO_FORMAT);
        return client.rank(scenario, 5);
    }
}