import React, { useState } from "react";
import {
  Box,
  Button,
  Input,
  InputGroup,
  InputRightElement,
  Text,
  VStack,
} from "@chakra-ui/react";

interface TaskManagerProps {
  onSend: (payload: { prompt: string; task_type?: string }) => void;
  response: string;
}

const TaskManager: React.FC<TaskManagerProps> = ({ onSend, response }) => {
  const [prompt, setPrompt] = useState("");

  const handleSend = () => {
    if (prompt.trim() !== "") {
      onSend({ prompt }); // ✅ correct key
      setPrompt("");
    }
  };

  return (
    <VStack spacing={4} align="stretch" mt={6}>
      <Box>
        <InputGroup>
          <Input
            placeholder="Type your command..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSend();
            }}
            bg="whiteAlpha.100"
            color="white"
          />
          <InputRightElement width="4.5rem">
            <Button h="1.75rem" size="sm" onClick={handleSend} colorScheme="teal">
              Send
            </Button>
          </InputRightElement>
        </InputGroup>
      </Box>
      <Box>
        <Text fontSize="sm" color="gray.400">
          Response:
        </Text>
        <Text fontSize="md" color="white">
          {response}
        </Text>
      </Box>
    </VStack>
  );
};

export default TaskManager;

