import pygame, sys, random


class Game:

    def __init__(self, agent):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.agent = agent
        # screen settings
        self.screen_width = 1280
        self.screen_height = 820

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        pygame.display.set_caption("Pong with agent")

        # game objects
        self.ball = pygame.Rect(self.screen_width / 2 - 15, self.screen_height / 2 - 15, 30, 30)
        self.player = pygame.Rect(self.screen_width - 10, self.screen_height / 2 - 70, 10, 140)
        self.opponent = pygame.Rect(10, self.screen_height / 2 - 70, 10, 140)

        self.bg_color = pygame.Color("grey12")
        self.light_grey = (200, 200, 200)

        self.ball_speed_x = 7 * random.choice((1, -1))
        self.ball_speed_y = 7 * random.choice((1, -1))
        self.player_speed = 0
        self.opponent_speed = 7

        # text variables
        self.player_score = 0
        self.opponent_score = 0

        self.game_font = pygame.font.Font("freesansbold.ttf", 32)

    def ball_animation(self):

        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y
        if self.ball.top <= 0 or self.ball.bottom >= self.screen_height:
            self.ball_speed_y *= -1
        if self.ball.left <= 0 or self.ball.right >= self.screen_width:
            if self.ball.left <= 0:
                self.player_score += 1
            if self.ball.right >= self.screen_width:
                self.opponent_score += 1
            self.agent.setState(self.getStateKey())
            self.ball_restart()

        # Collison
        if self.ball.colliderect(self.player) or self.ball.colliderect(self.opponent):
            self.ball_speed_x *= -1
            if self.ball.colliderect(self.player):
                self.agent.setState(self.getStateKey())

    def player_mov(self):
        for event in pygame.event.get():
            # keys event down and up
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.player_speed += 6
                if event.key == pygame.K_UP:
                    self.player_speed -= 6

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    self.player_speed -= 6
                if event.key == pygame.K_UP:
                    self.player_speed += 6

    def player_animation(self):
        self.player.y += self.player_speed
        if self.player.top <= 0:
            self.player.top = 0
        if self.player.bottom >= self.screen_height:
            self.player.bottom = self.screen_height

    def get_available_player_actions(self):
        if self.player.top == 0:
            return "DOWN"
        if self.player.bottom == self.screen_height:
            return "UP"
        else:
            return ["UP", "DOWN"]

    def opponent_ai(self):

        if self.opponent.top < self.ball.y:
            self.opponent.top += self.opponent_speed

        if self.opponent.bottom > self.ball.y:
            self.opponent.top -= self.opponent_speed

        if self.opponent.top <= 0:
            self.opponent.top = 0

        if self.opponent.bottom >= self.screen_height:
            self.opponent.bottom -= self.screen_height

    def ball_restart(self):
        self.ball.center = (self.screen_width / 2, self.screen_height / 2)
        self.ball_speed_y *= random.choice((1, -1))
        self.ball_speed_y *= random.choice((1, -1))

    def play(self, episodes):

        self.agent.train_agent(episodes)
        print("qtable after training(before game) : ", self.agent.qTable)
        # game loop
        while True:
            print(self.getStateKey())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # keys event down and up
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        self.player_speed += 6
                    if event.key == pygame.K_UP:
                        self.player_speed -= 6

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        self.player_speed -= 6
                    if event.key == pygame.K_UP:
                        self.player_speed += 6

            action = self.agent.get_action(self.getStateKey())
            print("agent game action is : ", action)

            if action == "UP":
                self.player_speed -= 6
            elif action == "DOWN":
                self.player_speed += 6

            # inject animation and logic
            self.ball_animation()
            self.player_animation()
            self.opponent_ai()

            # visuals
            self.screen.fill(self.bg_color)
            pygame.draw.rect(self.screen, self.light_grey, self.player)
            pygame.draw.rect(self.screen, self.light_grey, self.opponent)
            pygame.draw.ellipse(self.screen, self.light_grey, self.ball)
            pygame.draw.aaline(self.screen, self.light_grey, (self.screen_width / 2, 0),
                               (self.screen_width / 2, self.screen_height))
            player_txt = self.game_font.render(f"{self.player_score}", False, self.light_grey)
            self.screen.blit(player_txt, (600, 470))
            opponent_txt = self.game_font.render(f"{self.opponent_score}", False, self.light_grey)
            self.screen.blit(opponent_txt, (660, 470))
            # updating the window
            pygame.display.flip()
            self.clock.tick(60)

    def getStateKey(self):
        if self.ball.left <= 0 or self.ball.right >= self.screen_width:
            if self.ball.left <= 0:
                return "WIN"
            if self.ball.right >= self.screen_width:
                return "LOOSE"
        if self.ball.colliderect(self.player):
            return "TOUCHBALL"
        if self.player.bottom >= self.ball.y:
            return "BOTTOM"
        if self.player.top <= self.ball.y:
            return "TOP"

        return "DEFAULT"

    def move(self, action):
        if action == "UP":
            self.player_speed += 6
        if action == "DOWN":
            self.player_speed -= 6
        pass

    def getAgent(self):
        return self.agent

    def getBall(self):
        return self.ball

    def qlearn_agent_learn(self):

        pass
